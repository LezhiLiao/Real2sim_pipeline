# Third-Party Modules

This pipeline uses AprilTag/ArUco anchored 3DGS alignment as the scene/world alignment stage.

Recommended upstream reference:

- https://github.com/LezhiLiao/apriltag-3dgs-align

Use it as a submodule or external dependency when building the full project:

```bash
git submodule add https://github.com/LezhiLiao/apriltag-3dgs-align third_party/apriltag-3dgs-align
```

Do not mix its tag/camera coordinate convention directly with Isaac USD Camera convention. Tag detection and RealSense point clouds use an OpenCV-style optical frame, while Isaac Camera prims look along local `-Z`.
