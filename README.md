# smap_v2.1
smap_v2.1 installation and usage notes

v2.1 (25 Oct. 2024)

J. Peter Rickgauer

rickgauerj@janelia.hhmi.org

## OVERVIEW
smap_v2.1 is a program for detecting unlabeled macromolecules in 2D Cryo-EM images of cells that uses high-resolution template matching (HRTM; Rickgauer et al., eLife 2017; Rickgauer et al., BiorXiv 2020; Lucas et al., eLife 2021). High-resolution images are searched exhaustively to localize and align targets against an existing high-resolution structural model, and target validation is determined using a statistical approach that does not depend on the number of targets found in the image or on their similarities in an ensemble analysis, introducing new possibilities for detecting targets that are small or rare. Detected particles may be carried forward for further analysis, bearing in mind that most applications of conventional single-particle analysis (SPA) will yield reconstructions that resemble the template (Henderson, PNAS 2013; Van Heel, PNAS 2013); in such cases, excluding regions of novel structural interest from the search model before detection or further analysis is essential.

This repository includes source code, executables, and a containerized Docker environment to allow users to calculate new search models from solved high-resolution structures, and to employ those or other high-resolution .mrc-formatted structural volumes for target detection in pre-processed high-resolution 2D images of crowded molecular environments.


## SYSTEM RECOMMENDATIONS

-64-bit Ubuntu 22.04 (tested using 6.8.0-40-generic \#40~22.04.3-Ubuntu, x86_64)

-Minimal hardware configuration tested: Intel(R) Xeon(R) Gold 5218R CPU @ 2.10GHz (x80), 192 GB RAM, 3+ A5000 boards. Other NVIDIA-compatible GPU boards (GeForce GTX1080 or equivalent, with 8+ GB onboard memory) are also acceptable for most applications; runtime is reduced by maximizing single-precision FLOPS across boards (tested up to ~544 TFLOPS)

-Docker configured for the nvidia-container-toolkit


## INSTALLATION

Start by cloning the repository to your local machine:

```bash
git clone https://github.com/jpr-smap/smap_v2.1.git
cd smap_v2.1
```

The repository includes a Dockerfile to rebuild a containerized environment on Ubuntu 22.04 systems. Within that environment, our software can be run from a bash shell. 

We recommend starting from a clean install of Docker (version 27.3.1, build ce12230) and the nvidia-container-toolkit. If you are unfamiliar with Docker, the nvidia-container-toolkit, or both, please see the section on INSTALLATION HINTS. If your system *is* configured for this version of Docker but you are not sure whether the nvidia-container-toolkit is installed, you can check this with:
```bash
dpkg -l | grep nvidia-container-toolkit
```

If your system has a current installs of both Docker and the nvidia-container-toolkit, and compatible NVIDIA GPU boards, you should be able to rebuild the containerized environment directly from the cloned repo directory with:
```bash
sudo docker build -t smap:latest .
```
Rebuilding the environment for the first time may be slow (10s of minutes, depending on connectivity) because it involves downloading the full set of Matlab runtime environment files required to run the compiled software.

Once the environment is rebuilt, you should then be able to drop into it by running
```bash	
sudo docker run -it --gpus all smap:latest
```
to create a new interactive shell. You should land in the /opt/smap/ directory.



## SAMPLE RUN

The container environment's entry point (/opt/smap/) includes compiled executables and a sample .par file (sample_search.par), and the structure and image files needed to run a search based on this .par file. If the number of GPU boards you wish to allocate for this is fewer than the number present on the system, you can modify sample_search.par to change the value of [nCores] in the file (additional details about .par files are provided below). To start the sample search, you can run:
```bash	
./smap_run.sh sample_search.par
```
smap_run.sh is a simple bash script with one expected input argument (name_of_your_parfile.par). The script should indicate that the search has been launched, and it should then run in the background. To view the progress on this search, you can follow changes to files in the [outputDir] directory as specified in the sample_search.par file (by default, it is /opt/smap/result/061518_F_0012_cropped-6ek0_LSU/).

In the first step, a 3D .mrc file of the target molecule is calculated from the specified .cif file, which occurs in a temporary subdirectory of [outputDir]. This step takes about 4 minutes to complete for this LSU structure on our 4-board NVIDIA A5000 system. Once the work is complete and the model components are combined into two .mrc files representing electrostatic and scattering potential volumes (6ek0_LSU_EP.mrc and 6ek0_LSU_SP.mrc) in [outputDir], along with the combined log files, the temporary directory is removed.

In the second step, the scattering potential file volume is then used in an undersampled HRTM search of the image specified by [imageFile]. This step takes about 6 minutes to complete on our 4-board NVIDIA A5000 system for the cropped (864x864-pixel) image, and about 23 minutes for the full (3456x3456-pixel) image. Note that this search only samples a subset of the rotations needed to ensure complete coverage of orientation space at high resolution; a more typical search set will sample ~2.5x10^6 orientations (~8x the number tested here). To explore this crucial parameter, please see the section below on input file parameters. 

When completed, the [outputDir] for sample_search.par should include the following files:

	1. sample_search.par: a copy of the .par file used in the search
	2. searchImage.mrc: the image used for cross-correlation (pre-whitened if the [psdFilterFlag] input parameter is set to 1)
	3. search_SH.txt: For each specified SNR value (column 1), a list of survival histograms from flat-fielded search values determined in the search (column 3), or as expected for an equal number of values drawn from a Gaussian Normal distribution (column 2). Provides a survival histogram for the expected noise distribution (column 2 vs. column 1) and for the encountered SNR distribution for the target-search (column 3 vs. column 1)
	4. search_vals.mrc: maximum intensity projection (MIP) of all cross-correlation (CC) values above [arbThr] from each pixel in the image, after pixelwise flat-fielding (the input parameter [arbThr] is described below). Each section in the stack corresponds to one assumed defocus within the evenly-spaced range specified by the [df_inc] and [T_sample] input parameters
	5. search_qInds.mrc: indices of rotation matrices at each pixel in the MIP that produced the corresponding CC value. Multiple planes correspond to multiple assumed defocuses within the evenly-spaced range specified by the [df_inc] and [T_sample] input parameters
	6. search_mean.mrc: mean CC values at each pixel, calculated across all CCs in the search. Multiple planes correspond to multiple assumed defocuses within the evenly-spaced range specified by the [df_inc] and [T_sample] input parameters
	7. search_SD.mrc: standard deviation (SD) of all CC values at each pixel, calculated across all CCs in the search. Multiple planes correspond to multiple assumed defocuses within the evenly-spaced range specified by the [df_inc] and [T_sample] input parameters
	8. search_listAboveThreshold.dat: list of all CC values encountered during the search (prior to flatfielding, refinement, or optimization) above the threshold specified by the [arbThr] input parameter. Values are double-precision format and represent rotation index, a single coordinate location indexed against (x coordinate, y coordinate, defocus plane), and SNR value
	9. highVals.txt: list of initial (pre-flat fielded and -refined) values from the search exceeding the expected SNR cutoff threshold (=sqrt(2) * erfcinv(2/N_samples)). Sorted ascending by CC value. Columns indicate: [rotation_index CC coord1 coord2 defocus_index]
	10. search_SDs.dat: list of standard deviations (SD) for all templates tested in the search, prior to each template's normalization (i.e., setting the SD to one). Values are single-precision format and ordered ascending by searched rotation index
	11. list_final.txt: list of particles detected after flat-fielding, thresholding, and refinement. Columns indicate:
		CC	coord1	coord2	df1	df2	ang_ast	q1		q2		q3		q4
		13.338	585	121	410.70	288.90	0.960	0.437354	0.732569	-0.507308	0.121254
		13.182	589	587	453.70	331.90	0.960	0.592465	0.330980	-0.639164	-0.361810
		15.636	325	325	445.70	323.90	0.960	0.593458	-0.562495	0.529557	0.225778
		14.629	528	180	444.20	322.40	0.960	0.252096	0.734070	0.589035	-0.225006
		Parameters q1-q4 are elements of a normalized quaternion representing the rotation transform
	12. particles.mrc: 3D stack of detected particles from the image, cropped after sub-pixel centering on each detected particle
	13. templates.mrc: 3D stack of templates corresponding to the detected particles
	14. particles.par: a list of parameters for each detected particle from the search, provided in the format and convention of CisTEM/FrealignX:
	C           PSI   THETA     PHI       SHX       SHY     MAG  FILM      DF1      DF2  ANGAST     OCC      LogP      SIGMA   SCORE  CHANGE
	      1  -70.79  126.01   39.80      0.00      0.00   48450     1   4107.0   2889.0  -55.00  100.00         0       0.50    0.00    0.00
	      2    4.04   92.07   58.78      0.00      0.00   48450     1   4537.0   3319.0  -55.00  100.00         0       0.50    0.00    0.00
	      3  112.44  101.16 -154.10      0.00      0.00   48450     1   4457.0   3239.0  -55.00  100.00         0       0.50    0.00    0.00
	      4  -87.01  140.49  170.46      0.00      0.00   48450     1   4442.0   3224.0  -55.00  100.00         0       0.50    0.00    0.00
	15. particles.txt: final SNR values for detected particles after optimization
	16. particles.mat: an output file with a MATLAB-readable table that lists detected particles’ final estimated sub-pixel centered coordinates, alignments (as rotation matrices), and SNR values after optimization
	17. search.log: consolidated log file from search step
	18. 6ek0_LSU_EP.mrc: electrostatic potential volume calculated for the search target here
	19. 6ek0_LSU_SP.mrc: scattering potential volume calculated for the search target here
	20. 6ek0_LSU.log: consolidated log file from target potential calculations


## INPUT FILE PARAMETERS

The sample_search.par file included in the landing directory includes descriptions (commented, following a \# sign) for input parameters and values expected in [your_parfile.par]. Any line beginning with a \# sign is ignored.


	# [function]: the compiled function to run

	function search_global

	# [nCores]: number of GPU boards to request:

	nCores 4

	# [imageFile]: name of input image to search (.mrc). The input image should already be pre-processed (i.e., corrected for gain reference, motion-corrected by frame, and summed to form a single-frame .mrc file)

	imageFile /opt/smap/image/061518_F_0012_cropped.mrc

	# [modelFile]: name of scattering potential volume to use for template generation (.cif file, as formatted in the examples within the models/ directory, or .mrc). If you provide an .mrc file (e.g., a scattering potential calculated from a previous search), it will use that file as the target instead of calculating a new one

	modelFile /opt/smap/model/6ek0_LSU.cif

	#modelFile /opt/smap/model/5j5b_monster.pdb

	# [bFactor]: assumed B-factor for all atoms if a new scattering potential is being calculated. Defaults to 0 if unlisted, and ignored if <modelFile> is a preexisting .mrc file

	bFactor 0

	# [outputDir]: directory for output and scratch files

	outputDir /opt/smap/result/061518_F_0012_cropped-6ek0_LSU

	# [aPerPix]: voxel or pitch assumed for the input image and model (in Angstroms)

	aPerPix 1.032

	# [defocus]: astigmatic defocus parameters for the image (units: angstroms, angstroms, degrees) (see Rohou and Grigorieff, JSB 2015)

	defocus 4407.0 3189.0 -55.0

	# search specs:

	# [aPerPix_search]: pixel-pitch assumed for the search. If [aPerPix_search] differs from [aPerPix], the image and scattering potential are resampled by a factor of [aPerPix]/[aPerPix_search] for the global search and refinement steps; for the final step (particle optimization), the original non-resampled image and SP are used

	aPerPix_search 1.5

	# [rotationsFile] or [angle_inc]: two options to specify the set of rotations tested in the search. If [angle_inc] is used, a custom rotations file (rotations.txt) is written to the output directory during an early stage of the search.

	# [rotationsFile] is an ASCII file (space-delimited) with a list of indexed 3x3 rotation matrices to employ during the search. Each 3x3 rotation matrix, R, included in the file should be normalized. 

	# [angle_inc] specifies the average spacing between out-of-plane or in-plane rotations to search (you can additionally specify [psi_inc] as a new line of the .par file if you wish to provide a separate increment for in-plane rotations). Note that a typical high-resolution search with a ~3 A structure uses increments of ~1.88 degrees, increasing the runtime by ~8-fold.

	angle_inc 3.8

	#rotationsFile /opt/smap/rotation/hopf_R3.txt

	# [T_sample]: estimated sample thickness (units: nanometers). Used together with [df_inc] to determine the range of assumed defocus planes to search

	T_sample 200

	# [df_inc]: defocus step-size used in the global image search (units: nanometers) 

	df_inc 50

	# microscope properties:

	# [V_acc]: microscope accelerating voltage (units: volts)

	V_acc 300000.0

	# [Cs]: spherical aberration coefficient (units: meters)

	Cs 0.000001

	# [Cc]: chromatic aberration coefficient (units: meters)

	Cc 0.0027

	# [deltaE]: energy spread of the source (units: eV)

	deltaE 0.7

	# [a_i]: illumination aperture (units: radians)

	a_i 0.000050

	# optimization specs:

	# [optThr]: minimum SNR (pre-flat fielded) needed to qualify a particle (cluster) for post-search refinement and optimization

	optThr 7.0

	# [qThr]: minimum angular distance (units: degrees) separating two above-threshold CC maxima included in a cluster

	qThr 10

	# [dThr]: minimum euclidean distance (units: Angstroms) separating two above-threshold CC maxima include in a cluster

	dThr 10

	# [range_degrees]: angular range searched during refinement (units: degrees). If the [angle_inc] parameter is passed to the global search, range_degrees is automatically set to [angle_inc]

	range_degrees 2.0

	# [inc_degrees]: angular increment searched during refinement (units: degrees)

	inc_degrees 0.5

	# optional parameters:	

	# [arbThr]: threshold CC value above which all values are saved (with corresponding pixel coordinates and rotation matrix indices). Values smaller than 6.0 may be explored but will slow down the search, and rapidly increase storage and memory demands

	arbThr 6.0

	# [keep_scratch_flag]: debugging flag that determines whether intermediate files in the scratch subdirectory are kept or deleted at the conclusion of a search

	keep_scratch_flag 0

	# [margin_pix]: Determines the margin near the image-edges in which CC values found during the search are excluded from refinement or optimization. Intended to minimize residual edge artifacts from camera artifacts

	margin_pix 32 


## FILES INCLUDED

Dockerfile: used to rebuild a containerized environment on Ubuntu 22.04 systems allowing executables in the package to be invoked from a bash shell

smap_run.sh: bash script to start a global search using the Python implementation. Syntax to run is `./smap_run.sh <your_parfile.par>`

run_smappoi.py: Python wrapper that reads the requested function from a parameter file and runs the corresponding module

sample_search.par: sample parameter file

image/061518_F_0012_[cropped or full].mrc: sample images to search, excerpted from mouse embryonic fibroblast cells imaged close to focus (~300 nm).

model/6ek0_LSU.cif: large ribosomal subdomain excerpted from the original structure (Natchiar et al., Nature 2017) that can be used as a search model

model/5j5b_monster.pdb: rearranged ~2 MDa bacterial ribosome structure that can be used as a control search (Cocozaki et al., PNAS 2016).

rotation/hopf_R3.txt: sample [rotationsFile] entry, illustrating the format. Note that this provides only partial rotation coverage for a high-resolution search

README.md: this readme file


## Usage

To run a search using the Python implementation inside Docker, execute `smap_run.sh` with a parameter file:

```bash
./smap_run.sh sample_search.par
```

This script reads the `function` entry from the parameter file and dispatches to the matching module, such as

```bash
python -m smap_tools_python.smappoi_search_global sample_search.par 1
```

For single-run invocations outside the Docker wrapper you can also call:

```bash
python run_smappoi.py sample_search.par
```

`smap_run.sh` uses the same approach to launch one process per GPU based on the `nCores` value in the parameter file.


## INSTALLATION HINTS

The following worked for us to install both Docker and the nvidia-container-toolkit on several Ubuntu 22.04 systems.

Begin by checking to see if an older version of Docker is installed, and if so, uninstall it:
```bash
sudo apt-get remove docker docker-engine docker.io containerd runc 
```
Next, set up the Docker repository:
```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
```
Add the Docker repo:
```bash
echo \
"deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
$(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list ] /dev/null
```
Install the Docker engine:
```bash
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```
If this worked, you can verify the Docker install:
```bash
sudo docker --version
```
If your Docker does not already include the nvidia-container-toolkit, you may be able to install it by first setting up the NVIDIA package repository:
```bash
sudo curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```
Try installing by running: 
```bash
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```
If this doesn’t work, try this:
```bash
sudo apt-key adv --fetch-keys https://nvidia.github.io/nvidia-docker/gpgkey
```
Then try the install again:
```bash
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```
Then configure Docker for the toolkit:
```bash
sudo tee /etc/docker/daemon.json ] /dev/null <<EOF
{
    "runtimes": {
        "nvidia": {
       	    "path": "nvidia-container-runtime",
            "runtimeArgs": []
       }
    }
}
EOF
```
Restart Docker:
```bash
sudo systemctl restart docker
```
Verify that this has worked (it may also download the cuda/ubuntu base at this point):
```bash
sudo docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```
If this worked, it should show you the output from nvidia-smi (a list of the boards available on the system). 

It should then be possible to rebuild the containerized environment using:
```bash
sudo docker build -t smap:latest .
```
and to drop into the environment using:
```bash
sudo docker run -it --gpus all smap:latest
```

## KNOWN ISSUES

The parser for .pdb and .cif files has not been tested extensively on the range of possible .pdb and .cif file formats. If you encounter difficulties calculating scattering potentials from a .pdb or .cif-formatted file, please check whether the format matches one of the .cif or .pdb files provided in the model/ directory.


