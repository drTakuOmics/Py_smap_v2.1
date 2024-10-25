classdef smap < handle % % placeholder for now
    properties (SetObservable = false, GetAccess = 'public')
        prefs
    end
    
    methods
        function obj=smap(varargin)
            
        end
    end
        
    methods (Static)

        % image file I/O:
        outref=ri(fn,varargin);
        [outref,rez]=mr(filename,startSlice,numSlices,varargin);
        outref=tr(fn);
        mw(map,filename,rez);
        tw(m,fn,bps);
        [map,s,mi,ma,av]=ReadMRC(filename,startSlice,numSlices,varargin);
        WriteMRC(map,rez,filename);
        outref=WriteMRCHeader(map,rez,filename,nim);
        [m,sx,units]=ReadDMFile(filename,logfile);
        dw(inref,fn,varargin);
        varargout=read_cif_file(params_cif,varargin);
        numref=FOV_to_num(FOVref);
        FOVref=num_to_FOV(numref);
        
        [paramsref,fn_type]=readParamsFile(fnref,varargin);
        indsref=assignJobs(nIndices,nServers,serverID,varargin);
        R=readRotationsFile(inref,varargin);
        writeRotationsFile(inref,varargin);
        [ss,qBest,sI,xy]=clusterImByThr(imRef,mipiRef,thr,rotationsQ,varargin);
        outref=normalizeRM(inputRM,varargin);
        varargout=read_pdb_file(fn,varargin);
        [a,b] = parameterizeSF(varargin);
        [ai,al,av]=readDatFile(fn,varargin);
        
        % image processing:
        mask=makeFilt(imSize,maskEdges,aPerPix,varargin);
        [outref,mask,D]=mask_a_volume(mapref,mask_params,varargin);
        outref = mask_central_cross(imref,varargin);
        outref=ccf(imref,templateref);
        [ccOut,peaks]=ccff(imref,templateref,varargin);
        outref=ccfn(imref,templateref);
        outref=ccfv(imref,templateref);
        outref=ipcc(imref,templateref);
        [filterOut,imOut,psbg]=psdFilter(nfIm,varargin);
        psd=getPSD(nfIm,varargin);
        imOut=applyFilter(imref,tm,varargin);
        [k_2d,centerPixel]=getKs(imref,aPerPix);
        outref=rrj(inref);
        outref=getcp(inref);
        [nxny,peakVal,errorFlags]=maxInterpF(imref,varargin);
        outref=cropPatchFromImage3(inref,halfDim,rowColInds);
        outref=applyPhaseShifts(inref,shifts,varargin);
        outref=resize_F(inref,sf,varargin);
        outref=cutj(inref,varargin);
        outref=extendj(inref,varargin);
        outref=iftj(inref);
        outref=ftj(inref);
        [outref,shifts,peak]=reg2vols(inputVol,refVol,varargin);
        outref=projView(volref);
        outref=radialmeanIm(imref,varargin);
        [outref,outref_2d]=radialmeanj(imref,varargin);
        outref=particleDiameter(volref,varargin);
        mask=cosMask(imSize,maskEdges,aPerPix);
        outref=padForFFT(varargin);
        outref=rTheta(inref);
        [r,g,b] = polarImage(A, Ar, Ac, Nrho, Ntheta, Method, Center, Shape);
        volref_out=rotate3dMatrix(volref,R,varargin);
        imref_out=rotate2dMatrix(imref,R,varargin);
        outref=radialmaxj(imref,varargin);
        outref=doseFilter(imref,totalDose,aPerPix,varargin);
        filt_out=estimate_detector(dirref,varargin);
        F=MTF_MM(params,xdata);
        [outref,dummy_noW,otherref]=backproject(patchref,Rref,varargin);
        filtref=psdFilter_3d(inref,varargin);
        [ym,yb,y_full] = bindata(y,x,xrg,varargin);
        occref_out=occ(imref,templateref,varargin);
        outref=rot90j(inref,varargin);
        [V_s,V_ctrl_s,qOut,qOut_grid,inds,V_inds]=q_to_density(qA,qB,varargin);
        SNR=estimate_SNR(MW_here,thickness,varargin);
        [outref,scaled_template]=subtractVolume(recref,templateref,varargin);
        outref=cropOrPad(inref,varargin);
        outref=resizeForFFT(inref,varargin);
        gOut=g2(XYZ,varargin);
        outref=dust(volref,criteria,varargin);

        % model generation:
        outref=searchForPDB(varargin);
        varargout=PDB2EP(structureName,nmPerVoxel,varargin);
        outref=EP2SP(structureName,edgeSize,varargin);
        outref=templates(obj,qref,dfs,varargin);
        outref=templates_gpu(obj,qref,dfs,varargin);
        outref=templates_half_gpu(obj,qref,dfs,varargin);
        [outref,shifts]=register_multiple_fragments(fn_array,varargin);

        % dataset pre-processing:
        obj=gainCorr(obj);
        obj=motionCorr(obj,varargin);
        obj=sumFrames(obj,varargin);
        obj=runCTFFind(obj);
        obj=preprocess(obj);
        obj_out=getDots(obj,varargin);
        
        % search post-processing:
        varargout=readOutputFiles(objOrID,fileTypeOrPath);        
        [ti,templateIm]=makeTemplateStack(nfIm,varargin);%templates,ss);

        % electron microscope simulation:
        CTF=ctf(df,edgeSize,pixelSize,varargin);
        MTFout=approxMTF(sfs,inputParams,varargin); % remove extra sfs input
        ppOut=makePhasePlate(imref,varargin);

        % numerical:
        v_out=rotate3dVector(R,v);
        outref=mean(inref);
        outref=nm(inref);
        qOut=bumpQ(qIn,qBump);
        distInDeg=measureQD(q1,q2,varargin);
        distInDeg_array=pairwiseQD(q1,q2);
        [r,qOut]=frealign2smap(ori);
        [ori,qOut]=smap2frealign(r);
        axan=smap2pymol(RMref,varargin);
        outref=def_consts();
        [outref_RM, outref_EA] = calculate_search_grid(symmetry_symbol, angular_step_size, psi_step, varargin);
        outref=q2R(inref,varargin);
        q_ref=gridded_qs(range,inc,varargin);
        [thr_ref,prec,recall,thr_F1,F1_opt]=PR_quick(tableref1,varargin);
        rrs_degref=LB_BH_to_RRS(q_LBref,q_BHref,varargin);


        % plot/display tools:
        [xs,ys,YS]=plotSH(peakVals,varargin);
        qFig(fname,varargin);
        thr=plotSHH(bins,N,varargin); 
        avl(varargin);
        ahl(varargin);
        p3do(xyzref,varargin);
        p3d(xyzref,varargin);
        p3dr(xyzref,varargin);
        p3a(varargin);
        outref=tileImages(imstack);
                
        % utilities (glue):
        outref=checkBaseDir(inref);
        outref=whoami(varargin);
        varargout=gpuwhos;
        out=ts;
        outref=getPref(handles,pref);
        smapProps=getDatasets(filename,varargin);
        s=getDataset(datasetName,handles);
        putDataset(s,handles);
        [inds,entries]=parseCellArray(inref,str);
        outref=parseExcelFile(varargin);
        zpOut=zp(numIn,N,varargin);
        angles_out=smap2cistem(R_ref);
        angles_out=cistem2smap(EA_ref);
        gpu_whos(varargin);

        % overly specific:
        [qOut,xyz_sub,xyz_rnap]=getIcos(qBest,varargin);
        [xyz_sub,xyz_rnap]=icos(varargin);
        
        % % not yet incorporated:
        
%         spmd_test();
%         smappoi();
%         smappoi_monitor();
%         
%         PDB2SP();
%         qsub_PDB();


%         varargout=smappoi(fileref,objectFile);
%         varargout=parseInputFile(obj);
%         find2dCOM();
%         subtractTemplate();
%         particle_q_and_xyz();
%         poissrndj();
%         patchCC_fxn();
%         online_GC();
%         online_runCTFFind();
%         smap_online_test();
%         phaseFlip();
%         phaseScramble();

% % not required for deployment:
%         interp3gpu();
%         probe(datasetref,modelref,varargin);
%         qdel_script();
%         center();
%         rigid_transform_3D();
%         nm_F();
%         slidingK();
%         slidingK_noFilt();
%         getThresh();
%         normalizeRM();
%         pr();
%         prLines();
%         makeFilt();

%         
%         smap_pdb();
%         lcf();
%         fitcurve();
%         
%         smap_patchCC_collect_parts();
%         radialGeneralize();
%         getRes();
%         makeBumpQs();
%         optimizeMatch();

    end        
end

