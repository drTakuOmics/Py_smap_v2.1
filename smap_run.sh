#!/bin/bash
# script for running compiled smap code inside Docker
# Sets up the MATLAB Runtime environment for the current $ARCH and executes 
# the specified command.
# If the number of GPU boards requested in the .par file (nCores X) is greater than the number of boards visible to nvidia-smi -L, the lower of the two numbers is used
#
# syntax: ./smap_run.sh <parfile.par>

exe_name=$0
exe_dir=`dirname "$0"`
paramsFile=$1
echo ${paramsFile}

available_gpus=(`nvidia-smi -L | wc -l`)
nToRun=(`grep 'nCores' $paramsFile | grep '^[^#;]' | sed 's/^.* //'`)
if [ "$available_gpus" -lt "$nToRun" ]; then
    echo "Updating nCores in $paramsFile to $available_gpus (number of available GPUs)"
    
    # Update the nCores line in the .par file
    sed -i "s/^nCores.*/nCores $available_gpus/" "$paramsFile"

    # Set nCores to the new value
    nToRun=$available_gpus
else
    echo "nCores ($nToRun) is within the available GPU count ($available_gpus)."
fi

fxnToRun=(`grep 'function' $paramsFile | grep '^[^#;]' | sed 's/^.* //'`)

# Use the MCRROOT that is set in the Docker image environment
echo "function to run is $fxnToRun ($nToRun boards requested)"
echo "MATLAB Runtime libraries are located at ${MCRROOT}"

LD_LIBRARY_PATH=.:${MCRROOT}/runtime/glnxa64
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${MCRROOT}/bin/glnxa64
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${MCRROOT}/sys/os/glnxa64
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${MCRROOT}/sys/opengl/lib/glnxa64
export LD_LIBRARY_PATH
#echo LD_LIBRARY_PATH is ${LD_LIBRARY_PATH}

numDone=0
currentNum=1
while [[ $currentNum -le $nToRun ]]
do
    echo starting on process $currentNum ...
    echo "${exe_dir}/smappoi_$fxnToRun $paramsFile $currentNum"
    ${exe_dir}/smappoi_$fxnToRun $paramsFile $currentNum > /dev/null &
    ((currentNum=currentNum+1))
done
exit

