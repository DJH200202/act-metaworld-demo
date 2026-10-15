# Third-party notices

- This repository uses **Hugging Face LeRobot 0.6.1**, licensed under Apache-2.0. ACT's core policy, processors, dataset implementation, and training engine are provided by LeRobot. This project does not claim authorship of those components.
- Simulation and the privileged-state expert are provided by **MetaWorld 3.0.0**. See that package's license for its code and assets.
- The visual backbone starts from torchvision's **ResNet18 ImageNet1K V1** weights. The released inference checkpoint contains the resulting trained visual backbone. Refer to torchvision's documentation and the source dataset's applicable terms when reusing weights.
- PyTorch, torchvision, MuJoCo, Gymnasium, Accelerate, PyAV, and other dependencies retain their respective licenses. Installing this repository does not replace those licenses.
- Reported trajectories and videos were generated in simulation for this experiment. No physical robot or human demonstration recording was used.

The MIT license in this repository covers the original experiment orchestration and documentation authored for this project.
