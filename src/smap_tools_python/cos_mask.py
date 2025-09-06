diff --git a//dev/null b/src/smap_tools_python/cos_mask.py
index 0000000000000000000000000000000000000000..f0f568e63164b62ebb9cd1cb8458a107c252da5a 100644
--- a//dev/null
+++ b/src/smap_tools_python/cos_mask.py
@@ -0,0 +1,28 @@
+import numpy as np
+
+
+def rrj(shape):
+    """Compute normalized radial coordinates for a 2D grid."""
+    dim_max = max(shape)
+    cp = dim_max // 2
+    x = np.arange(dim_max) - cp
+    X = np.broadcast_to(x, (dim_max, dim_max))
+    Y = X.T
+    R = np.sqrt(X**2 + Y**2)
+    R = R / (2 * R[cp, 0])
+    return R[: shape[0], : shape[1]]
+
+
+def variable_cos_mask(im_size, mask_edges, a_per_pix):
+    """Replicates MATLAB's variableCosMask function."""
+    mask_edge_in, mask_edge_out = mask_edges
+    nn = rrj((im_size, im_size))
+    R = nn / a_per_pix
+    Rt = R <= mask_edge_in
+    RR = np.abs(R - mask_edge_in) * (~Rt)
+    Rtt = R > mask_edge_out
+    RR[Rtt] = np.pi / 2
+    T = (mask_edge_in - mask_edge_out) * 2
+    RRR = 0.5 + 0.5 * np.cos(2 * np.pi * RR / T)
+    RRR[Rtt] = 0
+    return RRR
