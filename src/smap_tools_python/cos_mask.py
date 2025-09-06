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
