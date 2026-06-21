# FashionMNIST post-hoc vs SENN notebook walkthrough

## Goal and context
This notebook is meant to help you understand how three explanation methods behave on the same
trained SENN model (FashionMNIST, robust_reg=1e-2, 5 concepts):
- SENN built-in explanations (concepts and relevances from the model itself)
- Integrated Gradients (post-hoc, gradient-based)
- LIME (post-hoc, perturbation-based)

The point is not just to draw pictures, but to answer a few practical questions:
- Do the explanations look sensible for the same image?
- When you remove what the method says is important, does confidence actually drop?
- How expensive is each method to run at inference time?

## Inputs and prerequisites
You need three things on disk before this notebook makes sense:
- The config (configs/fashion_mnist_lambda1e-2_c5_seed29.json), which tells the trainer what
  experiment name to use and how the model was set up.
- The trained checkpoint (results/fashion_mnist_lambda1e-2_c5_seed29/checkpoints/best_model.pt),
  which is the model you are explaining.
- The post-hoc artifacts for IG and LIME (results/.../posthoc/ig_* and lime_*). The notebook
  does not recompute those; it assumes you already ran the scripts to save time.

If those files are missing, the notebook stops early because it cannot line up predictions
and attributions.

## Step-by-step walkthrough
1. Imports and constants
   This is the setup phase. It loads libraries, defines the label names for FashionMNIST,
   and sets the key paths. Choosing the device here is important because everything that
   follows (model loading, ablations, plotting) relies on a consistent device.

2. Load SENN model
   The model is restored from the checkpoint and put in eval mode. Conceptually, this step
   gives you access to two things:
   - The usual logits/predictions.
   - The SENN internals (concepts h and relevances theta), which are what make SENN
     self-explaining.

3. Load precomputed IG and LIME results
   Instead of recomputing IG/LIME every time, the notebook loads the saved tensors and metrics.
   This is more about staying consistent: you want IG and LIME results computed once, then
   reused so all comparisons are on exactly the same data.

4. Select representative samples
   The notebook picks one example per class so that the qualitative plots cover the whole
   dataset. It then finds the matching indices in the IG/LIME saved outputs. This is the
   alignment step: without it, you could accidentally plot IG attributions for a different
   image than the one you are showing.

5. SENN built-in explanations
   Here you see how SENN explains itself. For each image, you get:
   - The raw image and predicted label.
   - A bar plot of relevances theta for the predicted class.
   - A bar plot of concept activations h.
   The goal is to see what concepts are active and which ones the model is actually using
   to drive its prediction.

6. Integrated Gradients saliency maps
   IG is a pixel-space explanation: it tells you which pixels push the prediction up or down.
   The notebook turns the saved attribution tensor into a heatmap so you can visually compare
   it with the image.

7. LIME saliency maps
   LIME produces a similar type of output (pixel importance), but via perturbation rather than
   gradients. The same visualization style is used so the visual comparison is fair.

8. Side-by-side comparison
   This is the “put everything in one view” section. For each class sample, you see the image,
   the SENN relevance bars, and both IG and LIME heatmaps. This makes it easier to spot cases
   where methods agree, disagree, or highlight very different regions.

9. Computational cost
   This section answers “how expensive is it?” It measures how long SENN takes to produce
   explanations (basically one forward pass), then compares that to IG and LIME using their
   saved timing metadata. The log-scale chart emphasizes how much more expensive post-hoc
   methods are per sample.

10. Faithfulness via relative ablation (Top vs Random)
   The idea: if a method says some features are important, removing them should reduce model
   confidence more than removing random features. Because pixel masking can create OOD images,
   the notebook uses a relative drop (Top minus Random) to cancel that artifact.
   It tests this both for IG/LIME (top 20 percent pixels) and for SENN (top concept by
   theta-only and by signed h*theta).

11. Faithfulness via Spearman correlation
   This is a more “ranking-aware” test. Instead of just top vs random, it looks at whether
   the ranking of features matches the ranking of actual drops.
   - SENN: correlate concept importance with the drop from ablating each concept.
   - IG/LIME: split pixels into five bins by |attribution|, ablate each bin, and correlate
     average attribution with the observed drop.
   - Random baseline: do the same with shuffled attributions to show what “no signal” looks like.

12. Discussion and final summary table
   The discussion connects the dots: why signed h*theta is used for SENN, what the qualitative
   plots show, and how cost compares. The summary table packages all the key numbers into one
   place so you can quickly compare methods.

13. Extra analyses ("nuova aggiunta di claude")
   This block digs into two practical concerns:
   - Is zeroing a concept actually fair, or does it push the model OOD? The fill=mean test
     checks that.
   - Are concepts h and relevances theta correlated in practice? The correlation plot gives
     a sense of whether those two signals are redundant or complementary.

## Results produced when you run the notebook
You get a mix of plots and printed summaries that tell the story:
- Visual comparisons (SENN bars, IG/LIME heatmaps, and the 4-row side-by-side grid).
- Cost comparisons (table + log-scale bar chart).
- Faithfulness metrics (relative drop bars, distributions, and Spearman rho charts).
- Final tables that gather the core metrics.

## Potential issues to check (serious)
- The Spearman comparison bar chart cell uses senn_spearman, which is not defined in this
  notebook. That cell will raise a NameError unless you define senn_spearman
  (for example, select one SENN variant or explicitly aggregate them).
- The final summary table uses all_senn_drop_top and all_senn_drop_relative, which are aliased
  to the theta-only variant earlier. If you intend to report the h*theta signed variant as
  your main SENN faithfulness metric, update those fields to avoid a mismatch.
