# Implementation choices

TE-Fed implements the manuscript method with the executable interfaces below. Where the manuscript leaves an interface unspecified, this repository defines the choice explicitly.

[METHOD_CONTRACT.md](METHOD_CONTRACT.md) records the method invariants: source expert identities and weights stay fixed during verification, cases become eligible after their creation round, missing active fidelity measurements reject a case, and aggregation retains the mean-reliability factor.

| Interface | Implementation |
| --- | --- |
| Detector | Ultralytics 8.4.0 YOLO26s; one shared trunk/neck, one general head and five specialist heads. |
| Multimodal fusion | A trainable residual input adapter maps normalized RGB/depth/NIR to the RGB trunk. Single-modality views use the same adapter with missing inputs masked. |
| Specialist architecture | Configurable projection width compatible with the native Ultralytics criterion; each run records measured parameter counts. |
| Router | A normalized linear projection maps 17 cues/state values to a five-dimensional prototype space. The general prototype is the normalized all-ones vector; specialists use coordinate prototypes. |
| Correction route | Recorded source head identities train their current student counterparts and stay fixed during local presentations. |
| Detection criterion | Each active head uses native assignment/loss. [TRAINING.md](TRAINING.md) defines head activation and target weighting. |
| Cross-modal alignment | Aligned RoIAlign and a shared trainable projection; negatives span the local minibatch. Sensor-specific forwards retain parameter gradients with frozen BN running statistics. |
| Weak/strong consistency | One-to-one same-class matches at IoU >= 0.50; equally weighted categorical KL and GIoU terms on differentiably decoded one-to-many predictions. Photometric strong views preserve coordinates. |
| Checkpoints | Model schema 2 includes the projection. Earlier checkpoints require regenerated pretraining and calibration. |
| Baselines | [BASELINES.md](BASELINES.md) defines the detector adaptations, teacher operators, penalties and aggregation rules used by the shared runner. |
| Initialization | Optional supervised FedAvg pretraining uses local labeled partitions. Comparators share one initial checkpoint. |
| Calibration | A validation artifact measures image-space probes and priors; [CALIBRATION.md](CALIBRATION.md) defines the probe conditions. Its hash is saved with the run. Test and hard labels are excluded. |
| Evaluation seeds | Configurations default to seed IDs 0--9; summaries require the declared seed set. |
| Rendering assets | Local `.blend` files are external; the repository includes rendering and asset-inspection code. |
| NIR rendering | Metadata identifies synthetic intensity as a proxy when spectral materials are unavailable. This is not a physical NIR sensor simulation. |

[TEST_REPORT.md](TEST_REPORT.md) records executed software checks. Dataset metrics and device resources require the corresponding completed training and measurement runs.
