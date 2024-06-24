# Hyperbolic Random Forests

This is the official repository for the paper [Hyperbolic Random Forests](https://openreview.net/pdf?id=pjKcIzvXWR).

### Abstract: 
Hyperbolic space is becoming a popular choice for representing data due to the hierarchical structure - whether implicit or explicit - of many real-world datasets. Along with it comes a need for algorithms capable of solving fundamental tasks, such as classification, in hyperbolic space. Recently, multiple papers have investigated hyperbolic alternatives to hyperplane-based classifiers, such as logistic regression and SVMs. While effective, these approaches struggle with more complex hierarchical data. We, therefore, propose to generalize the well-known random forests to hyperbolic space. We do this by redefining the notion of a split using horospheres. Since finding the globally optimal split is computationally intractable, we find candidate horospheres through a large-margin classifier. To make hyperbolic random forests work on multi-class data and imbalanced experiments, we furthermore outline a new method for combining classes based on their lowest common ancestor and a class-balanced version of the large-margin loss. Experiments on standard and new benchmarks show that our approach outperforms both conventional random forest algorithms and recent hyperbolic classifiers.

<img src="assets/fig1.png" width="430" height="353" />

## Data

The network embeddings can be downloaded from [this repository](https://github.com/hhcho/hyplinear). We provide the WordNet embeddings and labels for our experiments [here](https://drive.google.com/drive/folders/14Mmp_jGmLu5jkKpvv_vIR7K-e0Pdl8BV?usp=sharing).

## Acknowledgements
- We make use of the [Hyperbolic Hierarchical Clustering](https://github.com/HazyResearch/HypHC) and [Hyperbolic Image Embeddings](https://github.com/leymir/hyperbolic-image-embeddings) repositories.

## Citation
If you find our work relevant to your research, please cite:
```
@article{
  doorenbos2024hyperbolic,
  title={Hyperbolic Random Forests},
  author={Lars Doorenbos and Pablo M{\'a}rquez Neila and Raphael Sznitman and Pascal Mettes},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2024},
  url={https://openreview.net/forum?id=pjKcIzvXWR},
  note={}
}
```

## License
The code is published under the [MIT License](LICENSE).
