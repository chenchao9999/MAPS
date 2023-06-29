# MAPS
MAPS：for the zero-shot entity linking

## Data Preparation:

create a **data** fold;

linking data;
prepare the ZESHEL and place it under the fold **data**; [ZESHEL](https://github.com/facebookresearch/BLINK)

typing data;
Prepare the ultra-fine entity typing data from <https://www.cs.utexas.edu/~eunsol/html_pages/open_entity.html>, place it under **data**

## first stage: entity typing for labels
1. Encoding type and its description: **python typing/encode_types.py**
2. Train the ultra-fine entity typing model: **python typing/main.py**
3. Generate type information of the zero-shot entity linking data: **python typing/main.py --mode generate --load_model "./models/berttype"**
4. Merge all type information: **python generate_types.py**

## second multi-task learning
1. Before entity linking , do the anchor point sampling;
using the src label or type label to the selection; 
and then generate the negative mention-entity pairs;
using different **train_K_neg** in the **blink** fold;

2. Run zero-shot entity linking candatate generation: following <https://github.com/facebookresearch/BLINK>



