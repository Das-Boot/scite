# SCITE <img src="https://raw.githubusercontent.com/Das-Boot/scite/master/scite.png" align="right" width="561px">
***S***elf-Attentive BiLSTM-***C***RF w***I***th with ***T***ransferred ***E***mbeddings for Causality Extraction

[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.neucom.2020.08.078-blue)](https://doi.org/10.1016/j.neucom.2020.08.078)
[![Twitter URL](https://img.shields.io/twitter/url?label=%40lizhn7&style=social&url=https%3A%2F%2Ftwitter.com%2Flizhn7)](https://twitter.com/lizhn7)

**Code and data for： <br />**
**Li, Z., Li, Q., Zou, X., & Ren, J. (2021). Causality extraction based on self-attentive BiLSTM-CRF with transferred embeddings.** *Neurocomputing*. <br />
[DOI: 10.1016/j.neucom.2020.08.078](https://doi.org/10.1016/j.neucom.2020.08.078).
(Table 6 in this version does not appear to have been edited correctly; please see [arXiv](https://arxiv.org/abs/1904.07629) for a correctly formatted paper.)
___

## Highlights
- A novel causality tagging scheme has been proposed to serve the causality extraction
- Transferred embeddings dramatically alleviate the problem of data insuﬃciency
- The self-attention mechanism can capture long-range dependencies between causalities
- Experimental results show that the proposed method outperforms other baselines
___

## Abstract

Causality extraction from natural language texts is a challenging open problem in artificial intelligence. Existing methods utilize patterns, constraints, and machine learning techniques to extract causality, heavily depending on domain knowledge and requiring considerable human effort and time for feature engineering. In this paper, we formulate causality extraction as a sequence labeling problem based on a novel causality tagging scheme. On this basis, we propose a neural causality extractor with the BiLSTM-CRF model as the backbone, named SCITE (Self-attentive BiLSTM-CRF wIth Transferred Embeddings), which can directly extract cause and effect without extracting candidate causal pairs and identifying their relations separately. To address the problem of data insufficiency, we transfer contextual string embeddings, also known as Flair embeddings, which are trained on a large corpus in our task. In addition, to improve the performance of causality extraction, we introduce a multihead self-attention mechanism into SCITE to learn the dependencies between causal words. We evaluate our method on a public dataset, and experimental results demonstrate that our method achieves significant and consistent improvement compared to baselines.
___

## Keywords

Causality extraction, Sequence labeling, BiLSTM-CRF, Flair embeddings, Self-attention
___

## Download link for the model logs

- Baidu Netdisk: https://pan.baidu.com/s/18CfLFFQ3IRPGLX8QDJdSlg password: v5c2
- Google Drive: https://drive.google.com/file/d/1qJRYF3RZ2Fc4kLSXHZ5weNgC11BjWDGK/view?usp=sharing
___

## Citation
Please cite the following paper when using SCITE.
    
    @article{Li2021,
      author = {Li, Zhaoning and Li, Qi and Zou, Xiaotian and Ren, Jiangtao},
      doi = {10.1016/j.neucom.2020.08.078},
      URL = {http://www.sciencedirect.com/science/article/pii/S0925231220316027},
      issn = {18728286},
      journal = {Neurocomputing},
      pages = {207-219},
      title = {Causality extraction based on self-attentive BiLSTM-CRF with transferred embeddings},
      volume = {423},
      year = {2021}
    }
___

For bug reports, please contact Zhaoning Li ([yc17319@umac.mo](mailto:yc17319@umac.mo), or [@lizhn7](https://twitter.com/lizhn7)).

Thanks to [shields.io](https://shields.io/).
___

## LICENSE

<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>, which gives you the right to re-use and adapt, as long as you note any changes you made, and provide a link to the original source.
