import torch
import torch.nn as nn
import json
import dataclasses
from typing import Any, Union
from trl import GRPOTrainer
from trl.models import unwrap_model_for_generation
from trl.trainer.grpo_trainer import selective_log_softmax
import transformers
from accelerate.utils import gather
from transformers import DataCollatorForSeq2Seq
from latent_grpo_processor import CFEnhancedLogitsProcessor
from torch.distributed import all_reduce, all_gather, ReduceOp
from transformers.trainer import TrainOutput, TrainerState
from transformers import TrainingArguments

def swap_adjacent_blocks(x, k):
    original_shape = x.shape
    x_2d = x.view(-1, k)
    n = x_2d.size(0)
    indices = torch.arange(n).view(-1, 2).flip(1).reshape(-1)
    return x_2d[indices].view(original_shape)

class CustomTrainerState(TrainerState):
    def save_to_json(self, json_path: str):
        """保存状态为 JSON 文件，同时处理张量"""
        state_dict = dataclasses.asdict(self)
        # 转换张量为可序列化格式
        for key in list(state_dict.keys()):
            if isinstance(state_dict[key], torch.Tensor):
                state_dict[key] = state_dict[key].tolist()
            elif isinstance(state_dict[key], list) and any(isinstance(i, torch.Tensor) for i in state_dict[key]):
                state_dict[key] = [i.tolist() if isinstance(i, torch.Tensor) else i for i in state_dict[key]]
        
        # 确保 log_history 可序列化
        if "log_history" in state_dict:
            for log in state_dict["log_history"]:
                for k, v in log.items():
                    if isinstance(v, torch.Tensor):
                        log[k] = v.item()
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(state_dict, f, indent=2, sort_keys=True)

class NoiseGRPORecTrainer(GRPOTrainer):
    def __init__(self, prefix_allowed_tokens_fn, hgpo_weight=0.5, num_groups=3, 
                 fairness_weight=0.1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hgpo_weight = hgpo_weight 
        self.num_groups = num_groups 
        self.fairness_weight = fairness_weight  # 公平性正则项权重
        
        # 初始化组统计信息
        self.global_group_rewards = [[] for _ in range(num_groups)]
        self.global_group_counts = torch.zeros(num_groups, dtype=torch.long, device=self.accelerator.device)
        self.global_group_means = torch.zeros(num_groups, dtype=torch.float32, device=self.accelerator.device)
        
        # 初始化 l_harm 为0（将在第一个epoch后更新）
        self.l_harm = torch.tensor(0.0, device=self.accelerator.device)
        self.prev_l_harm = torch.tensor(0.0, device=self.accelerator.device)  # 上一轮的l_harm
        
        # 存储epoch数据
        self.epoch_rewards = []
        self.epoch_groups = []
        self.epoch_updated = False  # 标志是否已更新epoch级统计
        
        # 设置数据整理器
        data_collator = DataCollatorForSeq2Seq(self.processing_class, pad_to_multiple_of=8, 
                                              return_tensors="pt", padding=True)
        def data_collate_fn(batch):
            new_batch = data_collator(batch)
            new_batch['group'] = torch.tensor([item['group'] for item in batch], dtype=torch.long)
            return new_batch
        self.data_collator = data_collate_fn
        
        # 设置生成配置
        self.prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self.generation_config = transformers.GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=False,
            temperature=self.generation_config.temperature,
            pad_token_id=self.processing_class.pad_token_id,
        )
        
        # 使用自定义的TrainerState
        self.state = CustomTrainerState()
        self.state.global_step = 0
        self.state.epoch = 0.0  # 初始化为0.0
        self.state.max_steps = 0
        self.state.num_train_epochs = self.args.num_train_epochs
        self.state.log_history = []
        
    def save_state(self):
        """保存训练状态，确保可序列化"""
        # 保存前将张量转换为可序列化格式
        if hasattr(self, 'global_group_counts'):
            self.state.global_group_counts = self.global_group_counts.tolist()
        if hasattr(self, 'global_group_means'):
            self.state.global_group_means = self.global_group_means.tolist()
        if hasattr(self, 'l_harm'):
            self.state.l_harm = self.l_harm.item()
        if hasattr(self, 'prev_l_harm'):
            self.state.prev_l_harm = self.prev_l_harm.item()
        
        super().save_state()
        
    def load_state(self):
        """加载训练状态"""
        super().load_state()
        # 恢复张量状态
        if hasattr(self.state, 'global_group_counts'):
            self.global_group_counts = torch.tensor(self.state.global_group_counts, 
                                                   device=self.accelerator.device)
        if hasattr(self.state, 'global_group_means'):
            self.global_group_means = torch.tensor(self.state.global_group_means, 
                                                   device=self.accelerator.device)
        if hasattr(self.state, 'l_harm'):
            self.l_harm = torch.tensor(self.state.l_harm, device=self.accelerator.device)
        if hasattr(self.state, 'prev_l_harm'):
            self.prev_l_harm = torch.tensor(self.state.prev_l_harm, device=self.accelerator.device)

    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
        logits = logits[:, :-1, :]
        input_ids = input_ids[:, -logits_to_keep:]
        logits = logits[:, -logits_to_keep:]
        logits = logits / self.temperature
        return selective_log_softmax(logits, input_ids)

    def my_get_per_token_logps(self, model, input_ids, inputs_embeds, attention_mask, logits_to_keep):
        if hasattr(model, 'module'):
            model = model.module
        model_to_call = model.module if hasattr(model, 'module') else model
        outputs = model_to_call(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            use_cache=False,
            logits_to_keep=logits_to_keep + 1
        )
        logits = outputs.logits
        logits = logits[:, :-1, :]
        input_ids = input_ids[:, -logits_to_keep:]
        logits = logits[:, -logits_to_keep:]
        logits = logits / self.temperature
        return selective_log_softmax(logits, input_ids)

    @torch.no_grad()
    def _ppl_calculation(self, model, input_ids, attention_mask, input_embeds, logits_to_keep):
        per_token_logps = self.my_get_per_token_logps(
            model, input_ids, input_embeds, attention_mask, logits_to_keep
        )
        return (-per_token_logps.detach().clone().sum(dim=-1)/logits_to_keep).exp()

    def _generate_and_score_completions(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        prompt_inputs = super(GRPOTrainer, self)._prepare_inputs(inputs)
        groups = prompt_inputs['group']
        prompt_completion_ids, prompt_completion_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        labels = prompt_inputs["labels"]
        labels_length = (labels[0] != -100).sum(dim=-1)
        prompt_length = prompt_completion_ids.size(1) - labels_length
        prompt_ids = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]
        completion_mask = prompt_completion_mask[:, prompt_length:]
        prompt_mask = prompt_completion_mask[:, :prompt_length]
        logits_to_keep = completion_ids.size(1)
        with torch.no_grad():
            batch_size = prompt_completion_ids.size(0)
            original_embeds = self.model.generate_embs(prompt_completion_ids, prompt_completion_mask)
            where_thought_ids = torch.nonzero(
                prompt_completion_ids == self.model.model.embed_tokens.num_embeddings - 1
            )
            noise = torch.randn(
                (batch_size, original_embeds.size(-1)), device=self.model.device
            ).mul(1.5)
            noise[0, :] = 0
            original_embeds[torch.arange(batch_size), where_thought_ids[:, 1]] += noise
        with torch.no_grad():
            old_per_token_logps = self.my_get_per_token_logps(
                self.model, prompt_completion_ids, original_embeds, prompt_completion_mask, logits_to_keep
            )
            if self.beta == 0.0:
                ref_per_token_logps = None
            else:
                ref_per_token_logps = self.my_get_per_token_logps(
                    self.ref_model, prompt_completion_ids, original_embeds, prompt_completion_mask, logits_to_keep
                )
        prompts = [None for _ in range(len(prompt_ids))]
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            output_reward_func_test = -(-old_per_token_logps.clone().sum(dim=-1)/labels_length).exp().to(torch.float32)
            rewards_per_func[:, i] = output_reward_func_test
        rewards_per_func = gather(rewards_per_func)
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)
        rewards_gathered = gather(rewards)
        mean_grouped_rewards_orig = rewards_gathered.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards_orig = rewards_gathered.view(-1, self.num_generations).std(dim=1)
        mean_grouped_rewards_orig = mean_grouped_rewards_orig.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards_orig = std_grouped_rewards_orig.repeat_interleave(self.num_generations, dim=0)
        temp_rewards = rewards_gathered.clone().view(-1, self.num_generations)
        xx = temp_rewards[:, 0].unsqueeze(1).expand_as(temp_rewards).reshape(-1)
        xx = swap_adjacent_blocks(xx, self.num_generations)
        advantages_orig = rewards_gathered - xx.mean()
        advantages_orig = advantages_orig / (torch.norm(advantages_orig) + 1e-6)
        groups_gathered = gather(groups)
        expected_group_size = rewards_gathered.size(0) // self.num_generations
        if groups_gathered.size(0) != expected_group_size:
            groups_gathered = groups_gathered[:expected_group_size]
        groups_gathered = groups_gathered.repeat_interleave(self.num_generations, dim=0)
        self.epoch_rewards.append(rewards_gathered.cpu())
        self.epoch_groups.append(groups_gathered.cpu())
        group_mean_dict = {g: rewards_gathered[groups_gathered == g].mean() if (groups_gathered == g).sum() > 0 else 0.0 for g in range(self.num_groups)}
        group_std_dict = {g: rewards_gathered[groups_gathered == g].std() if (groups_gathered == g).sum() > 0 else 1.0 for g in range(self.num_groups)}
        mean_grouped_rewards_hgpo = torch.tensor([group_mean_dict[g.item()] for g in groups_gathered], device=device)
        std_grouped_rewards_hgpo = torch.tensor([group_std_dict[g.item()] for g in groups_gathered], device=device)
        mean_grouped_rewards_hgpo = mean_grouped_rewards_hgpo[:rewards_gathered.size(0)]
        std_grouped_rewards_hgpo = std_grouped_rewards_hgpo[:rewards_gathered.size(0)]
        advantages_hgpo = (rewards_gathered - mean_grouped_rewards_hgpo) / (std_grouped_rewards_hgpo + 1e-4)
        advantages = self.hgpo_weight * advantages_hgpo + (1.0 - self.hgpo_weight) * advantages_orig
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]
        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "original_embeds": original_embeds,
            "noise": noise,
            "groups": groups,
        }

    def on_epoch_end(self, args, state, control, **kwargs):
        device = self.accelerator.device
        
        # 保存上一轮的l_harm作为历史记录
        self.prev_l_harm = self.l_harm.clone().detach()
        
        # 收集当前epoch的所有奖励和组信息
        all_epoch_rewards = []
        all_epoch_groups = []
        for rewards, groups in zip(self.epoch_rewards, self.epoch_groups):
            all_epoch_rewards.append(rewards.to(device))
            all_epoch_groups.append(groups.to(device))
        
        all_epoch_rewards = torch.cat(all_epoch_rewards) if all_epoch_rewards else torch.tensor([], device=device)
        all_epoch_groups = torch.cat(all_epoch_groups) if all_epoch_groups else torch.tensor([], device=device, dtype=torch.long)
        
        # 跨进程聚合数据
        gathered_rewards = [torch.zeros_like(all_epoch_rewards) for _ in range(torch.distributed.get_world_size())]
        gathered_groups = [torch.zeros_like(all_epoch_groups, dtype=torch.long) for _ in range(torch.distributed.get_world_size())]
        
        all_gather(gathered_rewards, all_epoch_rewards)
        all_gather(gathered_groups, all_epoch_groups)
        
        all_epoch_rewards = torch.cat(gathered_rewards)
        all_epoch_groups = torch.cat(gathered_groups)
        
        # 初始化组统计量
        group_sums = torch.zeros(self.num_groups, device=device)
        group_counts = torch.zeros(self.num_groups, device=device, dtype=torch.long)
        
        # 计算每个组的总奖励和样本数
        for g in range(self.num_groups):
            mask = (all_epoch_groups == g)
            if mask.any():
                group_sums[g] = all_epoch_rewards[mask].sum()
                group_counts[g] = mask.sum()
        
        # 跨进程同步总奖励和样本数
        all_reduce(group_sums, op=ReduceOp.SUM)
        all_reduce(group_counts, op=ReduceOp.SUM)
        
        # 计算全局组均值
        global_group_means = torch.zeros(self.num_groups, device=device)
        valid_groups = group_counts > 0
        global_group_means[valid_groups] = group_sums[valid_groups] / group_counts[valid_groups].float()
        
        # 计算组间方差 (l_harm)
        if valid_groups.sum() > 1:
            # 只考虑有样本的组
            valid_means = global_group_means[valid_groups]
            
            # 计算组间方差
            group_variance = valid_means.var()
            
            # 应用缩放因子并限制范围
            self.l_harm = 0.1 * group_variance
            self.l_harm = torch.clamp(self.l_harm, max=0.5)
        else:
            self.l_harm = torch.tensor(0.0, device=device)
        
        # 打印统计信息
        print(f"Epoch {state.epoch}: global_group_means: {global_group_means.tolist()}, l_harm: {self.l_harm.item()}, prev_l_harm: {self.prev_l_harm.item()}")
        
        # 重置epoch数据
        self.epoch_rewards = []
        self.epoch_groups = []
        self.epoch_updated = True

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        
        # 第一轮只收集数据，不进行梯度更新
        if not self.epoch_updated:
            # 创建虚拟损失（需要梯度但值为零）
            return torch.tensor(0.0, device=self.accelerator.device, requires_grad=True)
        
        # 正常训练流程
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        original_embeds = inputs["original_embeds"]
        noise = inputs["noise"]
        embeds = self.model.generate_embs(input_ids, attention_mask)
        where_thought_ids = torch.nonzero(
            input_ids == self.model.model.embed_tokens.num_embeddings - 1
        )
        batch_size = input_ids.size(0)
        embeds[torch.arange(batch_size), where_thought_ids[:, 1]] += noise
        per_token_logps = self.my_get_per_token_logps(
            model, input_ids, embeds, attention_mask, logits_to_keep
        )
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        advantages = inputs["advantages"]
        old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()
        
        # 添加前一轮计算的l_harm作为正则项
        loss = loss + self.fairness_weight * self.prev_l_harm
        
        # 记录指标
        mode = "eval" if self.control.should_evaluate else "train"
        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())
        
        # 记录公平性正则项值
        self._metrics[mode]["fairness_reg"].append(self.fairness_weight * self.prev_l_harm.item())
        
        return loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        # 第一轮只收集数据，不进行梯度更新
        if not self.epoch_updated:
            # 创建虚拟损失（需要梯度但值为零）
            return torch.tensor(0.0, device=self.accelerator.device, requires_grad=True)
        
        # 正常训练流程
        with self.accelerator.accumulate(model):
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
            self.accelerator.backward(loss)
            grad_norm = None
            if self.accelerator.sync_gradients:
                grad_norm = self.accelerator.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                # 确保记录的是标量值而非张量
                grad_norm_value = grad_norm.item() if grad_norm is not None else 0.0
                self.log({"grad_norm": grad_norm_value})
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
        return loss.detach()

    def _inner_training_loop(self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None):
        from transformers.trainer_utils import get_last_checkpoint
        import math
        import os
        self._train_batch_size = batch_size
        self.args = args
        total_train_batch_size = args.per_device_train_batch_size * args.world_size * args.gradient_accumulation_steps
        completed_steps = 0
        starting_epoch = 0
        tr_loss = torch.tensor(0.0).to(self.accelerator.device)
        self._globalstep_last_logged = 0
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)
        train_dataloader = self.get_train_dataloader()
        total_steps = len(train_dataloader)
        num_update_steps_per_epoch = math.ceil(total_steps / args.gradient_accumulation_steps)
        max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
        self.create_optimizer_and_scheduler(num_training_steps=max_steps)
        metrics = {}  # 存储训练指标
        
        # 初始化prev_l_harm
        if not hasattr(self, 'prev_l_harm'):
            self.prev_l_harm = torch.tensor(0.0, device=self.accelerator.device)
        
        for epoch in range(starting_epoch, int(args.num_train_epochs)):
            # 关键修复：更新state.epoch为当前epoch（从1开始计数）
            self.state.epoch = epoch + 1.0
            
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)
            for step, inputs in enumerate(train_dataloader):
                if resume_from_checkpoint and step < self.state.global_step % total_steps:
                    continue
                
                tr_loss_step = self.training_step(self.model, inputs)
                
                # 处理虚拟损失
                if tr_loss_step is not None:
                    tr_loss += tr_loss_step
                
                if (step + 1) % args.gradient_accumulation_steps == 0 or step == total_steps - 1:
                    self.state.global_step += 1
                    completed_steps += 1
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                    
                    if args.logging_steps > 0 and completed_steps % args.logging_steps == 0:
                        avg_loss = tr_loss.item() / args.logging_steps
                        
                        # 计算当前epoch进度：state.epoch + (step / total_steps)
                        current_epoch_progress = self.state.epoch + (step / total_steps)
                        
                        logs = {
                            "loss": avg_loss, 
                            "epoch": current_epoch_progress,
                            "learning_rate": self.lr_scheduler.get_last_lr()[0],
                            "fairness_reg": self.fairness_weight * self.prev_l_harm.item(),
                            "l_harm": self.l_harm.item()
                        }
                        
                        metrics.update(logs)  # 更新 metrics
                        self.log(logs)
                        tr_loss.zero_()
                    
                    if args.save_steps > 0 and completed_steps % args.save_steps == 0:
                        self.save_model(os.path.join(args.output_dir, f"checkpoint-{completed_steps}"))
                    
                    if completed_steps >= max_steps:
                        break
            
            # 在每个epoch结束时更新l_harm
            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self.on_epoch_end(args, self.state, self.control)
            
            if completed_steps >= max_steps:
                break
        
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        return TrainOutput(global_step=completed_steps, training_loss=metrics.get("loss", 0.0), metrics=metrics)