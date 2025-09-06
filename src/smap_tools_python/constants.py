diff --git a//dev/null b/src/smap_tools_python/constants.py
index 0000000000000000000000000000000000000000..910c22eca5a15e3f6efdcb5194b5c1940c7fbad7 100644
--- a//dev/null
+++ b/src/smap_tools_python/constants.py
@@ -0,0 +1,22 @@
+import numpy as np
+
+
+def def_consts():
+    """Define physical constants used by SMAP tools."""
+    consts = {
+        "V": 300e3,
+        "m_e": 9.10938215e-31,
+        "h": 6.636e-34,
+    }
+    consts["hbar"] = consts["h"] / (2 * np.pi)
+    consts["q_e"] = 1.602e-19
+    consts["wl"] = 0.00197e-9
+    consts["IC"] = (2 * consts["m_e"] / (consts["hbar"] ** 2)) * consts["q_e"]
+    consts["k"] = 2 * np.pi / consts["wl"]
+    consts["Cs"] = 2.7e-3
+    consts["Cc"] = 2.7e-3
+    consts["a_i"] = 0.05e-3
+    consts["dE"] = 0.7
+    consts["c"] = 2.99792458e8
+    consts["c_v"] = consts["c"]
+    return consts
