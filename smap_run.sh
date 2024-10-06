#!/bin/bash
# script for execution of deployed applications inside Docker
#
# Sets up the MATLAB Runtime environment for the current $ARCH and executes 
# the specified command.
#

exe_name=$0
exe_dir=`dirname "$0"`
paramsFile=$1
echo ${exe_name}
echo ${paramsFile}

fxnToRun=(`grep 'function' $paramsFile | grep '^[^#;]' | sed 's/^.* //'`)
nToRun=(`grep 'nCores' $paramsFile | grep '^[^#;]' | sed 's/^.* //'`)

# Use the MCRROOT that is set in the Docker image environment
# The MCRROOT environment variable should already be defined, so no need to parse compute.cfg
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
    ##${exe_dir}/smappoi_$fxnToRun $paramsFile $currentNum
    ((currentNum=currentNum+1))
done
exit

