#   A Synergistic Framework for Learning Shape Estimation and Shape-Aware Whole-Body Control Policy for Continuum Robots
In this work, we present a novel synergistic framework for learning shape estimation and a shape-aware whole-body control policy for tendon driven continuum robots. Our approach leverages the interaction between two Augmented Neural Ordinary Differential Equations (ANODEs) - the **Shape-NODE** and **Control-NODE** - to achieve continuous shape estimation and shape-aware control. The *Shape-NODE* integrates prior knowledge from Cosserat rod theory, allowing it to adapt and account for model mismatches, while the *Control-NODE* uses this shape information to optimize a whole-body control policy, **trained in a Model Predictive Control (MPC) fashion**.
<div align="center">
  <img src="ctr_obs.gif" alt="Image description" width="600">
</div>
