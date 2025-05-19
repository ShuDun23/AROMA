# AROMA
A repo of [AROMA: Autonomous Rank-one Matrix Adaptation](http://arxiv.org/abs/2504.05343)

> **Abstract**:  
> As large language models continue to grow in size, parameter-efficient fine-tuning has become increasingly crucial. While low-rank adaptation (LoRA) offers a solution through low-rank updates, its static rank allocation may yield suboptimal results. Adaptive low-rank adaptation (AdaLoRA) improves this with dynamic allocation but remains sensitive to initial and target rank configurations. We introduce AROMA, a framework that automatically constructs layer-specific updates by iteratively building up rank-one components with very few trainable parameters that gradually diminish to zero. Unlike existing methods that employ rank reduction mechanisms, AROMA introduces a dual-loop architecture for rank growth. The inner loop extracts information from each rank-one subspace, while the outer loop determines the number of rank-one subspaces, i.e., the optimal rank. We reset optimizer states to maintain subspace independence. AROMA significantly reduces parameters compared to LoRA and AdaLoRA while achieving superior performance on natural language understanding and commonsense reasoning tasks, offering new insights into adaptive parameter-efficient fine-tuning.
<img src="https://github.com/ShuDun23/AROMA/blob/master/fig/workflow.jpg" width="500px">
<img src="https://github.com/ShuDun23/AROMA/blob/master/fig/MRPC.png" width="800px">

Our implementation is adapted from [ReLoRA](https://github.com/Guitaricet/relora).
