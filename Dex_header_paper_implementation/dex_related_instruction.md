It is **extremely common** for modern Android apps to have multiple DEX files (like `classes2.dex`, `classes3.dex`, etc.). In fact, for any non-trivial app developed in the last several years, having multiple DEX files is the standard, not the exception.

Here is why this happens and why it is a critical factor for your Vigidroid thesis project.

### Why do APKs have multiple DEX files?

The Dalvik Executable (DEX) specification has a hard architectural limit: a single `.dex` file can only reference **65,536 methods** (the "64K reference limit").

While 64K sounds like a lot, this limit includes:

* Methods written by the app developer.
* Android framework methods.
* Every method inside every third-party library the app uses.

Modern Android development relies heavily on massive libraries like Google Play Services, Firebase, Retrofit, and Jetpack Compose. Simply adding a few of these standard libraries easily pushes an app past the 64K limit.

To solve this, Android introduced **Multidex**. When the build system hits the 64K limit, it automatically overflows the compiled bytecode into `classes2.dex`, then `classes3.dex`, and so on. Furthermore, for any app with a Minimum SDK version of 21 (Android 5.0) or higher, multidex is enabled by default. Because Android 5.0 was released over a decade ago, essentially all modern apps use it.

---

### The Critical Flaw for Vigidroid

In the MSFDroid paper you shared earlier, the researchers state:

> *"This model mainly uses the classes.dex and AndroidManifest.xml files... so this paper ignores the other files in the APK file and only takes the classes.dex file"*

**This is a major blind spot.** If an app uses multidex, the primary `classes.dex` file often just contains the core startup classes and the multidex loader itself. The actual application logic—and potentially the **malicious payload**—could be sitting entirely in `classes2.dex` or `classes3.dex`. If Vigidroid only extracts `classes.dex`, a smart malware author could intentionally pad their app with junk methods to push their malicious code into a secondary DEX file, rendering your scanner blind to it.

### How to Fix This in Your Implementation

When building your extraction pipeline for Vigidroid, you cannot hardcode the extraction to just `classes.dex`. You must dynamically extract and process *all* DEX files.

Here is how you should update the Copilot implementation guideline for your thesis:

**Update to Task 1 & 2 (Feature Engineering):**
Instead of extracting just `classes.dex`, instruct Copilot to:

1. Use Python's `zipfile` module to iterate through the APK archive and find **all** files matching the regex pattern `classes.*\.dex`.
2. For the Dex Header branch: Parse the header of *every* DEX file found. You can handle the multiple headers in a few ways depending on your architecture:
* **Average/Sum Pooling:** Extract the 1D header tensor for `classes.dex`, `classes2.dex`, etc., and mathematically average them into a single 1D tensor before passing it to the neural network.
* **Concatenation (Fixed Size):** Support up to $N$ dex files (e.g., 3), pad with zeros if the app has fewer, and concatenate them.
* **Primary Focus:** If you must keep it incredibly lightweight, at minimum, sum the *sizes* of the identifiers across all DEX files to get a holistic view of the app's footprint, rather than just the footprint of the first file.



Handling multidex will make your Vigidroid implementation significantly more robust and practically applicable to real-world modern Android malware than the original paper!