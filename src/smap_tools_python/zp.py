+def zp(num_in, N, pad_char="0"):
+    """Zero-pad a number or string to a given width."""
+    s = str(num_in)
+    return s.rjust(N, pad_char)
