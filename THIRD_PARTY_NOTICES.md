# Third-party attribution and release terms

This project builds on the following upstream projects. Dependencies and model
weights retain their own licenses; this notice does not relicense them or grant
permissions on behalf of their owners.

- **TimeLens2**: MCG-NJU/TimeLens2, source revision
  `f828dde4ee7f2aea5022c6bd7f5aa1d42d4b3f35`;
  base model `MCG-NJU/TimeLens2-4B` at
  `ddbb6cb944f13ce21e59e85da23c5f356107260e`.
  The pinned source's `LICENSE` identifies Tencent copyright (2025), requires
  retention of notices, and includes academic-only and geographic restrictions.
  Its text is reproduced in `licenses/TimeLens2.txt`. Review the model repository
  and source terms before using or redistributing derived weights; this is not
  an assertion that the model has an unrestricted permissive license.
- **OMTG**: Tangent0308/OMTG, revision
  `3291c9e19490db19c7aae593791f80aaca6f53a8`; RL integration targets its vendored
  `rl/verl` subtree. Obtain the checkout separately with its original notices.
  The public launcher patches a private local copy, not the original checkout.
- **verl**, **Qwen3-VL**, **Hugging Face Transformers**, **PEFT**,
  **qwen-vl-utils**, **PyTorch**, **vLLM**, and **Ray** provide the training and
  inference foundations. Keep their distributions' respective licenses and
  copyright notices when packaging dependencies.

Code ownership/license selection and derivative-weight redistribution approval
remain maintainer release gates. No repository-wide MIT/Apache license or model
license metadata is selected automatically. Training/benchmark data require
separate permissions and are not redistributed with this code.
