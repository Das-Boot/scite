# SCITE
**S**elf-Attentive BiLSTM-**C**RF w**I**th with **T**ransferred **E**mbeddings for Causality Extraction.
- arXiv paper link: https://arxiv.org/abs/1904.07629
- free access link (before December 27, 2020): https://authors.elsevier.com/c/1c1ae3INukEDTx 
(Table 6 in the paper does not appear to have been edited correctly ...)

## Highlights
- A novel causality tagging scheme has been proposed to serve the causality extraction
- Transferred embeddings dramatically alleviate the problem of data insuﬃciency
- The self-attention mechanism can capture long-range dependencies between causalities
- Experimental results show that the proposed method outperforms other baselines

## Abstract

Causality extraction from natural language texts is a challenging open problem in artificial intelligence. Existing methods utilize patterns, constraints, and machine learning techniques to extract causality, heavily depending on domain knowledge and requiring considerable human effort and time for feature engineering. In this paper, we formulate causality extraction as a sequence labeling problem based on a novel causality tagging scheme. On this basis, we propose a neural causality extractor with the BiLSTM-CRF model as the backbone, named SCITE (Self-attentive BiLSTM-CRF wIth Transferred Embeddings), which can directly extract cause and effect without extracting candidate causal pairs and identifying their relations separately. To address the problem of data insufficiency, we transfer contextual string embeddings, also known as Flair embeddings, which are trained on a large corpus in our task. In addition, to improve the performance of causality extraction, we introduce a multihead self-attention mechanism into SCITE to learn the dependencies between causal words. We evaluate our method on a public dataset, and experimental results demonstrate that our method achieves significant and consistent improvement compared to baselines.

## Keywords

Causality extraction, Sequence labeling, BiLSTM-CRF, Flair embeddings, Self-attention

## Download link for the model logs

- Baidu Netdisk: https://pan.baidu.com/s/18CfLFFQ3IRPGLX8QDJdSlg password: v5c2
- Google Drive: 

## Citation
Please cite the following paper when using SCITE.

    @article{LI2021207,
      title = "Causality extraction based on self-attentive BiLSTM-CRF with transferred embeddings",
      journal = "Neurocomputing",
      volume = "423",
      pages = "207 - 219",
      year = "2021",
      issn = "0925-2312",
      doi = "https://doi.org/10.1016/j.neucom.2020.08.078",
      url = "http://www.sciencedirect.com/science/article/pii/S0925231220316027",
      author = "Zhaoning Li and Qi Li and Xiaotian Zou and Jiangtao Ren"
    }
