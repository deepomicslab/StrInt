# %% [markdown]
# # 0. Import libraries

# %%
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")
# import sys
# sys.path.insert(0, '/data6/wangjingwan/StrInt/')
from pyStrint.strint import strInt
from pyStrint import preprocess as pp
from pyStrint import plotting as pl

# %% [markdown]
# # 1. Load files

# %%
inputDir = './tutorial/demo/'
outDir = f'{inputDir}/results/'
if not os.path.exists(outDir):
    os.makedirs(outDir)
sc_exp = pd.read_csv(f'{inputDir}/SC_exp.tsv',sep = '\t',header=0,index_col=0)
sc_meta = pd.read_csv(f'{inputDir}/SC_meta.tsv',sep = '\t',header=0,index_col=0)
st_exp = pd.read_csv(f'{inputDir}/ST_exp.tsv',sep = '\t',header=0,index_col=0)
st_coord = pd.read_csv(f'{inputDir}/ST_coord.tsv',sep = '\t',header=0,index_col=0)
st_decon = pd.read_csv(f'{inputDir}/ST_weight.tsv',sep = '\t',header=0,index_col=0)
sc_distribution = pd.read_csv(f'{inputDir}/SC_smurf.tsv',sep = '\t',header=0,index_col=0)

# %% [markdown]
# # 2. Get cell model

# %%
import smurf
operator = smurf.SMURF(n_features=15, estimate_only=True)
sc_distribution = operator.smurf_impute(sc_exp.T).T

# %%
sc_distribution.to_csv(f'{outDir}SC_smurf.tsv',sep = '\t',header=True,index=True)

# %%
# or you can load the precomputed results
sc_distribution = pd.read_csv(f'{outDir}SC_smurf.tsv',sep = '\t',header=0,index_col=0)

# %% [markdown]
# # 3. Preprocess

# %%
species  = 'Mouse'
sc_adata, st_adata, sc_ref, lr_df = pp.prep_all_adata(sc_exp = sc_exp, st_exp = st_exp, sc_distribution = sc_distribution,
                                                    sc_meta = sc_meta, st_coord = st_coord, SP = species)

# %% [markdown]
# # 4. Cell selection

# %% [markdown]
# Cell selection parameters

# %%
st_tp = 'st'
num_per_spot = 7
repeat_penalty = int((st_exp.shape[0]*num_per_spot/sc_exp.shape[0]) * 10)
print('repeat_penalty:',repeat_penalty)

# %%
obj_spex = strInt(save_path = outDir, st_adata = st_adata, weight = st_decon,
                    sc_ref = sc_ref, sc_adata = sc_adata, cell_type_key = 'celltype', lr_df = lr_df,
                    st_tp = st_tp,species = species)
obj_spex.prep()

# %% [markdown]
# ### use strint for cell selection

# %%
sc_agg_meta = obj_spex.select_cells(p = 0, mean_num_per_spot = num_per_spot, repeat_penalty = repeat_penalty)

# %%
sc_agg_meta.to_csv(f'{outDir}/cell_mapping_meta.tsv',sep = '\t',header=True,index=True)

# %% [markdown]
# ### [or] load the precomputed results from strint or other methods
# 

# %%
sc_agg_meta = pd.read_csv(f'{outDir}/cell_mapping_meta.tsv',sep = '\t',header=0,index_col=0)
user_sc_exp = sc_exp.loc[sc_agg_meta['sc_id']]

# %%
sc_agg_meta = obj_spex.select_cells(user_sc_exp = user_sc_exp, user_sc_agg_meta = sc_agg_meta)

# %% [markdown]
# # 5. Refinement process

# %% [markdown]
# Gradient descent parameter recommandation 

# %%
delta, eta = [0.1, 0.0005]
max_rep = 30 # max_rep for gradient descent, choose accordingly.

# %%
expected_cell_num = st_adata.shape[0] *num_per_spot
ref_cell_num = sc_adata.shape[0]
lr_db = pp.load_lr_df(species = species) # full LR database
print(lr_db.shape)
print(lr_df.shape)

# %%
p1, p2, p3, p4 = pp.auto_tune_parameters(expected_cell_num/ref_cell_num, len(lr_df)/len(lr_db)) 
print(p1, p2, p3, p4)

# %%
refined_sc_exp, sc_agg_meta = obj_spex.gradient_descent(
                p1 = p1, p2 = p2, p3 = p3, p4 = p4, 
                delta = delta, eta = eta, 
                init_sc_embed = False,
                iteration = max_rep, k = 2, W_HVG = 2,
                left_range = 0, right_range = 5, steps = 1, dim = 2)
sc_agg_meta.to_csv(f'{outDir}/cell_mapping_meta.tsv',sep = '\t',header=True,index=True)

# %% [markdown]
# # 6. Plotting

# %%
species = 'Human'
new_exp_fn = f'{outDir}/refined_sc_exp.tsv'
new_meta_fn = f'{outDir}/cell_mapping_meta.tsv'

alter_exp = pd.read_csv(new_exp_fn,sep='\t',index_col=0,header=0)
sc_agg_meta = pd.read_csv(new_meta_fn,sep='\t',index_col=0,header=0)
refine_adata = pp.make_adata(alter_exp,sc_agg_meta,species,save_path = outDir, save_adata = False)
refine_adata.uns['tp_key'] = 'celltype'
refine_adata.uns['rscript_path'] = '/apps/software/R/4.2.0-foss-2021b/bin/Rscript'
refine_adata.uns['python_path'] = '/apps/software/Anaconda3/2022.05/bin/python'
# del alter_exp

# %%
pl.sc_celltype(refine_adata, color_map = None, tp_key = refine_adata.uns['tp_key'], figsize= (4,2),
        legend = True, size = 10, alpha = 1, savefig = True)


