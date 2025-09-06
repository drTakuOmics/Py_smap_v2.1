diff --git a//dev/null b/src/smap_tools_python/fov.py
index 0000000000000000000000000000000000000000..99dbbf906e779c5a635e96652a0419215756e3bf 100644
--- a//dev/null
+++ b/src/smap_tools_python/fov.py
@@ -0,0 +1,23 @@
+from datetime import datetime
+from .zp import zp
+
+ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
+
+
+def fov_to_num(fov_ref):
+    """Convert a field-of-view reference string to a numeric identifier."""
+    if isinstance(fov_ref, (list, tuple)):
+        fov_ref = fov_ref[0]
+    date_part, letter_part, num_part = fov_ref.split("_")
+    the_year = int("20" + date_part[4:6])
+    the_month = int(date_part[0:2])
+    the_date = int(date_part[2:4])
+    baseline = datetime(2014, 1, 1, 12, 0, 0)
+    target = datetime(the_year, the_month, the_date, 12, 0, 0)
+    days = (target - baseline).days
+    temp = list("0" * 9)
+    temp[0:4] = list(zp(days, 4))
+    idx = ALPHABET.index(letter_part.upper()) + 1
+    temp[4:6] = list(zp(idx, 2))
+    temp[6:9] = list(zp(int(num_part), 3))
+    return int("".join(temp))
