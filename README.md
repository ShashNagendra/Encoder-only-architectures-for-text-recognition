# Encoder-only-architectures-for-text-recognition
Implementation of the paper "Multi-Scale Encoder-Only Architectures for Enhanced Scene and Handwritten Text Recognition", submitted to The Visual Computer


 Install the required dependencies using: `pip install -r requirements.txt`







 **🔧 How to Run the STR/HTR Models**
 
To train or test any model, run the corresponding script from its folder:
<pre><code>```cd [folder_name] 
  ./train.ksh # For training 
  ./test.ksh # For testing ```</code></pre>



| **Folder**           | **Models**                                                                                     |
|----------------------|------------------------------------------------------------------------------------------------|
| `PVTSTR_V1`          | `pvt_tiny`, `pvt_small`, `pvt_medium`, `pvt_large`                                             |
| `SVTSTR_PCPVTSTR`    | `pcpvt_small_v0`, `pcpvt_base_v0`, `pcpvt_large_v0`, `alt_gvt_small`, `alt_gvt_base`, `alt_gvt_large` |
| `VAN`                | `van_b0`, `van_b1`, `van_b2`, `van_b3`, `van_b4`, `van_b5`, `van_b6`                           |



## **📦 Dataset Preparation**
This project follows the dataset structure and preparation method from the deep-text-recognition-benchmark by CLOVA AI.

**Option 1: Use Preprocessed LMDB Datasets**
Download ready-to-use LMDB datasets from the CLOVA benchmark:

📂 Directory structure:
<pre><code>``` data/
  └── data_lmdb_release/ 
  ├── training/
  └── evaluation/ ```</code></pre>
- 📎 [Download Links & Details](https://github.com/roatienza/deep-text-recognition-benchmark#download-data)


**Option 2: Create Your Own LMDB Dataset**
To use your own data, convert it to LMDB format using the `create_lmdb_dataset.py` script from the CLOVA repository:

<pre><code>```bash python3 create_lmdb_dataset.py \
  --input_path path/to/images \
  --gt_file path/to/labels.txt \ 
  --output_path data_lmdb_release/your_dataset ```</code></pre>
- 🔗 [Full instructions: CLOVA Deep Text Benchmark](https://github.com/clovaai/deep-text-recognition-benchmark)

## References
We have used the following papers and their official implementations as the foundation for our models and benchmarking:
- Atienza, Rowel. "Vision transformer for fast and efficient scene text recognition." *International Conference on Document Analysis and Recognition*, pp. 319–334. Springer, 2021.
- Wang, Wenhai et al. "Pyramid vision transformer: A versatile backbone for dense prediction without convolutions." *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pp. 568–578, 2021.
- Chu, Xiangxiang et al. "Twins: Revisiting the design of spatial attention in vision transformers." *Advances in Neural Information Processing Systems*, 34 (2021): 9355–9366.
- Guo, Meng-Hao et al. "Visual attention network." *Computational Visual Media*, 9(4), 733–752, 2023.







