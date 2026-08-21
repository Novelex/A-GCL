"""FROZEN UPSTREAM COMPATIBILITY COPY: the 08339b7 per-batch step, verbatim math.
Only Phase-1 execution repairs; imports production classes whose math S6/S8 proved
identical to 08339b7 defaults."""
import sys, torch, torch.nn.functional as F, numpy as np
sys.path.insert(0,"/users/3171356m/A-GCL")
from torch_scatter import scatter
def calc_regloss_upstream(z, aug, memory, temperature=0.1, pos_only=False):   # 08339b7 lines 23-46
    device=z.device; b=z.size(0)
    z=F.normalize(z,dim=-1); aug=F.normalize(aug,dim=-1); memory=F.normalize(memory,dim=-1)
    logits=torch.einsum("if, jf -> ij",z,aug)/temperature
    pos_mask=torch.zeros((b,b),dtype=torch.bool,device=device); pos_mask.fill_diagonal_(True)
    m_logits=torch.einsum("if, jf -> ij",z,memory)/temperature
    exp_logits=torch.exp(m_logits)
    log_prob=logits if pos_only else logits-torch.log(exp_logits.sum(1,keepdim=True))
    mean_log_prob_pos=(pos_mask*log_prob).sum(1)
    return -mean_log_prob_pos.mean()
class MemoryBank_Q_upstream:                                                   # 08339b7 lines 48-63
    def __init__(self,max_length,feature_dim,device):
        self.max_length=max_length
        self.memory=torch.zeros((max_length,feature_dim)).to(device)
        self.current_index=0
    def push(self,features,batch_size):
        features=features.detach()
        if (self.current_index+batch_size)<self.max_length:
            self.memory.data[self.current_index:self.current_index+batch_size,:]=features
            self.current_index=(self.current_index+batch_size)%self.max_length
        else:
            ci=batch_size-(self.max_length-self.current_index)
            self.memory.data[self.current_index:self.max_length,:]=features[:self.max_length-self.current_index,:]
            self.memory.data[0:ci,:]=features[self.max_length-self.current_index:,:]
            self.current_index=ci
def upstream_batch_step(args, model, view_learner, model_optimizer, view_optimizer,
                        memory_bank, batch, device, cap):
    """Verbatim 08339b7 training-loop body for one batch (lines ~150-235)."""
    view_learner.train(); view_learner.zero_grad(); model.eval()
    x,_=model(batch.batch,batch.x,batch.edge_index,None,batch.edge_weight)
    cap["x_view"]=x.detach().clone()
    edge_logits=view_learner(batch.batch,batch.x,batch.edge_index,None,batch.edge_weight)
    cap["edge_logits"]=edge_logits.detach().clone()
    temperature=1.0; bias=0.0+0.0001
    eps=(bias-(1-bias))*torch.rand(edge_logits.size(),device=device)+(1-bias)
    gate_inputs=torch.log(eps)-torch.log(1-eps)
    gate_inputs=(gate_inputs.to(device)+edge_logits)/temperature
    batch_aug_edge_weight=torch.sigmoid(gate_inputs).squeeze()
    cap["mask_view"]=batch_aug_edge_weight.detach().clone()
    x_aug,_=model(batch.batch,batch.x,batch.edge_index,None,batch_aug_edge_weight)   # mask REPLACES E
    cap["x_aug_view"]=x_aug.detach().clone()
    row,col=batch.edge_index; edge_batch=batch.batch[row]
    edge_drop_out_prob=1-batch_aug_edge_weight
    uni,edge_batch_num=edge_batch.unique(return_counts=True)
    sum_pe=scatter(edge_drop_out_prob,edge_batch,reduce="sum")
    reg=[]
    for b_id in range(args["batch_size"]):
        if b_id in uni:
            num_edges=edge_batch_num[uni.tolist().index(b_id)]
            reg.append(sum_pe[b_id]/num_edges)
    reg=torch.stack(reg).mean(); cap["reg"]=reg.detach().clone()
    memory_bank.push(x_aug, args["batch_size"])                                   # push BEFORE cr
    cap["queue_after_push"]=memory_bank.memory.detach().clone(); cap["queue_ptr"]=memory_bank.current_index
    cr_loss=calc_regloss_upstream(x,x_aug,memory_bank.memory); cap["cr_view"]=cr_loss.detach().clone()
    nce=model.calc_loss(x,x_aug); cap["nce_view"]=nce.detach().clone()             # defaults T=0.2 sym
    view_loss=nce-(args["reg_lambda"]*reg)+args["cr_lambda"]*cr_loss
    cap["view_loss"]=view_loss.detach().clone()
    (-view_loss).backward()
    cap["view_grads"]={n:p.grad.detach().clone() for n,p in view_learner.named_parameters() if p.grad is not None}
    view_optimizer.step()
    cap["view_params_after"]={n:p.detach().clone() for n,p in view_learner.named_parameters()}
    model.train(); view_learner.eval(); model.zero_grad()
    x,_=model(batch.batch,batch.x,batch.edge_index,None,batch.edge_weight)
    edge_logits=view_learner(batch.batch,batch.x,batch.edge_index,None,batch.edge_weight)
    eps=(bias-(1-bias))*torch.rand(edge_logits.size(),device=device)+(1-bias)
    gate_inputs=torch.log(eps)-torch.log(1-eps)
    gate_inputs=(gate_inputs.to(device)+edge_logits)/temperature
    batch_aug_edge_weight=torch.sigmoid(gate_inputs).squeeze().detach()
    x_aug,_=model(batch.batch,batch.x,batch.edge_index,None,batch_aug_edge_weight)
    cr_loss=calc_regloss_upstream(x,x_aug,memory_bank.memory); cap["cr_model"]=cr_loss.detach().clone()
    nce_m=model.calc_loss(x,x_aug); cap["nce_model"]=nce_m.detach().clone()
    model_loss=nce_m-args["cr_lambda"]*cr_loss                                     # ORIGINAL minus sign
    cap["model_loss"]=model_loss.detach().clone()
    model_loss.backward()
    cap["model_grads"]={n:p.grad.detach().clone() for n,p in model.named_parameters() if p.grad is not None}
    model_optimizer.step()
    cap["model_params_after"]={n:p.detach().clone() for n,p in model.named_parameters()}
    cap["opt_state"]=(len(model_optimizer.state), len(view_optimizer.state))
    return cap
