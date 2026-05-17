import os.path as path
from collections import defaultdict
import torch
import copy
from torch.utils.tensorboard import SummaryWriter
from transformers import BartTokenizer, BartForConditionalGeneration
import torch.nn.functional as F
import os
import policy_model
import field_loss

abs_path = path.abspath (__file__)
model_path = path.dirname (abs_path)
src_path = path.dirname (model_path)
base_path = path.dirname (src_path)
data_processed_path = path.join (base_path, "data_processed")
dot_before_path = path.join (data_processed_path, "dot_before")
dot_after_path = path.join (data_processed_path, "dot_after")
dot_file_path = path.join (dot_before_path, "0.dot")
black_list_path = path.join (base_path, "blacklist.txt")
parameter_path = path.join (model_path, "parameter")
pth_path = path.join (parameter_path, "model.pth")
runs_dir = path.join (base_path, "runs")
train_log_dir = path.join (runs_dir, "cell_exp")

black_list = set ()

def is_num (s) :
    return s.isdigit ()

class ContinuousField :
    def __init__ (self, latents : torch.Tensor, G : list[list[int]], device) :
        self.device = device
        self.max_n = 1024
        n, d = latents.shape
        self.adj_prob : torch.Tensor = torch.zeros ((self.max_n, self.max_n), device=self.device)
        self.alive_prob : torch.Tensor = torch.zeros (self.max_n, device=self.device)
        self.latents : torch.Tensor = torch.zeros ((self.max_n, d), device=self.device) 
        for i in range (n) :
            self.alive_prob[i] = 1.0
            self.latents[i] = latents[i]
            for j in G[i] :
                self.adj_prob[i, j] = 1.0

    def alive_num (self) :
        return self.alive_prob.sum ()

    def train (
        self, 
        adj_predict_policy : policy_model.AdjProbPredict,
        alive_predict_policy : policy_model.AliveProbPredict,
        latents_update_policy : policy_model.LatentPredict,
        correct_adj_prob : torch.Tensor,
        correct_alive_prob : torch.Tensor,
        correct_latents : torch.Tensor,
        adj_predict_optimizer,
        alive_predict_optimizer,
        latents_update_optimizer,
        current_epoch : int
    ) :
        rollout = 1
        for _ in range (rollout) :
            predict_adj_prob = adj_predict_policy (
                latents=self.latents,
                alive_prob=self.alive_prob,
                adj_prob=self.adj_prob
            )
            predict_alive_prob = alive_predict_policy (
                latents=self.latents,
                alive_prob=self.alive_prob,
                adj_prob=self.adj_prob
            )
            predict_latents = latents_update_policy (
                latents=self.latents,
                alive_prob=self.alive_prob,
                adj_prob=self.adj_prob
            )
            self.adj_prob = predict_adj_prob
            self.alive_prob = predict_alive_prob
            self.latents = predict_latents

        alive_loss = field_loss.alive_bce_loss (self.alive_prob, correct_alive_prob)
        latents_loss = field_loss.latents_loss (self.latents, correct_latents, correct_alive_prob)
        adj_probs_loss = field_loss.adj_prob_loss (self.adj_prob, correct_adj_prob, correct_alive_prob)
        topo_loss = field_loss.tree_topo_loss (self.adj_prob, self.alive_prob)
        total_loss = alive_loss + latents_loss + adj_probs_loss + topo_loss  
        print ("alive predict : ", self.alive_prob)
        print ("adj predict : \n", self.adj_prob)
        print ("correct alive : ", correct_alive_prob)
        print ("correct adj : \n", correct_adj_prob)
        print ("alive loss : ", alive_loss)
        print ("latents loss : ", latents_loss)
        print ("adj probs loss : ", adj_probs_loss)
        print ("topo_loss : ", topo_loss)
        print ("predict num : ", self.alive_num ())
        print ("expect num : ", correct_alive_prob.sum ())
        print ("total loss : ", total_loss)
        adj_predict_optimizer.zero_grad ()
        alive_predict_optimizer.zero_grad ()
        latents_update_optimizer.zero_grad ()
        total_loss.backward ()
        adj_predict_optimizer.step ()
        alive_predict_optimizer.step ()
        latents_update_optimizer.step ()

    def load_file (self, dot_path, tokenizer, model, device) :
        ast = AST (dot_path=dot_path)
        latents = []
        for node in ast.nodes :
            ast_kind_input = tokenizer (node.ast_kind, return_tensors="pt").to (device)
            with torch.no_grad () :
                ast_kind_latent = model.model.encoder (**ast_kind_input).last_hidden_state
            content_input = tokenizer (node.content, return_tensors="pt").to (device)
            with torch.no_grad () :
                content_latent = model.model.encoder (**content_input).last_hidden_state
            latent0 = ast_kind_latent.mean (dim=1, keepdim=True).to (device)  # (B, 1, d)
            latent1 = content_latent.mean (dim=1, keepdim=True).to (device)    # (B, 1, d)
            latent = torch.cat ([latent0, latent1], dim=2)   # (1, 1, 1536)
            latents.append (latent)
        latents = torch.stack (latents, dim=0)
        latents = latents.squeeze ()
        self.__init__ (
            latents=latents,
            G=ast.G,
            device=self.device
        )

class ASTNode :
    def __init__ (self, node_id : int, ast_kind : str,ast_id : int, content : str) :
        self.node_id = node_id
        self.ast_kind = ast_kind
        self.ast_id = ast_id
        self.content = content
    
    def print (self) :
        print (self.node_id, end=" ")
        print (self.ast_kind, end=" ")
        print (self.ast_id, end=" ")
        print (self.content)

class AST :
    def __init__ (self, dot_path : str) :
        self.n = 0
        self.nodes = []
        self.degs = []
        self.G = []
        self.G.append ([])
        with open (dot_path, "r") as f :
            for line in f :
                parts = line.strip ().split (maxsplit=3)
                if not parts :
                    continue
                node_id = int (parts[0])
                ast_kind = parts[1]
                if not is_num (ast_kind) :
                    ast_id = int (parts[2])
                    if len (parts) == 4 : 
                        content = parts[3]
                    else :
                        content = "None"
                    self.n += 1
                    self.nodes.append (ASTNode (node_id - 1, ast_kind, ast_id, content))
                    self.degs.append (0)
                    self.G.append ([])
                else :
                    self.G[node_id - 1].append (int (ast_kind) - 1)
                    self.G[int (ast_kind) - 1].append (node_id - 1)
                    self.degs[node_id - 1] += 1
                    self.degs[int (ast_kind) - 1] += 1

    def print (self) :
        print ("n = ", self.n)
        for i in range (0, self.n) :
            self.nodes[i].print ()
            print ("<", end="")
            for v in self.G[i] :
                print (v, end=" ")
            print (">")
    
    def print_degs (self) :
        print ("[", end="")
        for x in self.degs :
            print (x, end=" ")
        print ("]")
        

def main () :
    with open (black_list_path, "r") as f :
        for line in f :
            file = line.strip ()
            if file :
                black_list.add (file)
    writer = SummaryWriter (train_log_dir)
    device = torch.device ("cuda" if torch.cuda.is_available () else "cpu")
    print ("initialize bart.")
    tokenizer = BartTokenizer.from_pretrained ("facebook/bart-base")
    model = BartForConditionalGeneration.from_pretrained ("facebook/bart-base")
    model = model.to (device)
    print ("initialize success.")
    #ast_kind_pooler = attn_pool_project.AttnPoolProject (dim=768)
    #content_pooler = attn_pool_project.AttnPoolProject (dim=768)
    ast = AST (dot_file_path)
    print (type (dot_file_path))
    ast.print_degs ()
    latents = []
    print (type (ast.nodes))
    for node in ast.nodes :
        ast_kind_input = tokenizer (node.ast_kind, return_tensors="pt").to (device)
        with torch.no_grad () :
            ast_kind_latent = model.model.encoder (**ast_kind_input).last_hidden_state
        content_input = tokenizer (node.content, return_tensors="pt").to (device)
        with torch.no_grad () :
            content_latent = model.model.encoder (**content_input).last_hidden_state
        latent0 = ast_kind_latent.mean (dim=1, keepdim=True).to (device)  # (B, 1, d)
        latent1 = content_latent.mean (dim=1, keepdim=True).to (device)    # (B, 1, d)
        latent = torch.cat ([latent0, latent1], dim=2)   # (1, 1, 1536)
        latents.append (latent)
    latents = torch.stack (latents, dim=0)
    latents = latents.squeeze ()
    print ("latents shape : ", latents.shape)
    origin_feild = ContinuousField (
        latents=latents,
        G=ast.G,
        device=device
    )
    expect_feild = copy.deepcopy (origin_feild)
    # initialize cell graph and its spread tensor
    learning_rate = 1e-6
    epoch = 10
    round_times = 100
    adj_predict_policy = policy_model.AdjProbPredict (
        d_model=1536,
        d_hidden=2048,
        num_layers=6,
        topk=256
    ).to (device)
    adj_predict_optimizer = torch.optim.Adam (
        adj_predict_policy.parameters (),
        lr=learning_rate
    )
    alive_predict_policy = policy_model.AliveProbPredict (
        d_model=1536,
        d_hidden=2048,
        num_layers=6,
        topk=256
    ).to (device)
    alive_predict_optimizer = torch.optim.Adam (
        alive_predict_policy.parameters (),
        lr=learning_rate
    )
    latents_update_policy = policy_model.LatentPredict (
        d_model=1536,
        d_hidden=2048,
        num_layers=6,
        topk=256
    ).to (device)
    latents_update_optimizer = torch.optim.Adam (
        latents_update_policy.parameters (),
        lr=learning_rate
    )

    if path.exists (pth_path) :
        check_point = torch.load (pth_path, map_location=device)
        adj_predict_policy.load_state_dict (check_point["adj_predict"])
        alive_predict_policy.load_state_dict (check_point["alive_predict"])
        latents_update_policy.load_state_dict (check_point["latents_update"])

    # initialize the policy parameters

    train_tot = 0
    file_num = 0
    for i in range (epoch) :
        print ("epoch : ", end="")
        print (epoch)
        for file in sorted (os.listdir (dot_before_path), key=lambda x : int(x.split('.')[0])) :
            if file in black_list :
                continue
            try :
                #print ("GPU MB :", torch.cuda.memory_allocated () / 1024**2)
                print ("file name is ", end="")
                print (file)
                error_path = os.path.join (dot_before_path, file)
                right_path = os.path.join (dot_after_path, file)
                origin_feild.load_file (
                    dot_path=error_path,
                    tokenizer=tokenizer,
                    model=model,
                    device=device
                )
                expect_feild.load_file (
                    dot_path=right_path,
                    tokenizer=tokenizer,
                    model=model,
                    device=device
                )
                T = 1
                print ("current cell number is ", end="")
                print (origin_feild.alive_num (), end=", target cell number is ")
                print (expect_feild.alive_num ())
                for j in range (T) :
                    print ("T = ", j, end=" ")
                    #print ("GPU MB :", torch.cuda.memory_allocated () / 1024**2)
                    origin_feild.train (
                        adj_predict_policy=adj_predict_policy,
                        alive_predict_policy=alive_predict_policy,
                        latents_update_policy=latents_update_policy,
                        correct_adj_prob=expect_feild.adj_prob,
                        correct_alive_prob=expect_feild.alive_prob,
                        correct_latents=expect_feild.latents,
                        adj_predict_optimizer=adj_predict_optimizer,
                        alive_predict_optimizer=alive_predict_optimizer,
                        latents_update_optimizer=latents_update_optimizer,
                        current_epoch=i
                    )
                file_num += 1
            except Exception as e :
                print (f"[CRASH] {file}: {e}")
                with open (black_list_path, "a") as f :
                    f.write (file + "\n")
                black_list.add (file)
                continue
            train_tot += 1
            if train_tot % round_times == 0 :
                torch.save ({
                    "adj_predict" : adj_predict_policy.state_dict (),
                    "alive_predict" : alive_predict_policy.state_dict (),
                    "latents_update" : latents_update_policy.state_dict ()
                }, pth_path)

    return 0

if __name__ == '__main__' :
    main ()
