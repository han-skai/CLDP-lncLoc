### CLDP-lncLoc

CLDP-lncLoc is an lncRNA multi-label subcellular localization prediction model that utilizes cell line-specific single-cell expression profiles to construct heterogeneous networks. This approach enables the exploration of potential mechanisms influencing lncRNA subcellular localization from the perspectives of cell type and gene expression levels. Additionally, CLDP-lncLoc employs contrastive loss to optimize the representations of prototypes and samples in the latent space. Furthermore, the model incorporates a dynamic threshold module to independently learn classification thresholds for each subcellular location, effectively addressing challenges posed by imbalanced datasets.

### Environmental requirements

```python
python == 3.7.12
torch == 1.13.1 
dgl == 1.1.2 
numpy  == 1.21.6
pandas == 1.3.5
scikit-learn == 1.0.2
```