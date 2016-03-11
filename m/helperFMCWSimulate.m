function [xr,xr_unmixed] = helperFMCWSimulate(Nsweep,hwav,hradarplatform,hcarplatform,...
    htx,hchannel,hcar,hrx)
% This function helperFMCWSimulate is only in support of FMCWExample. It
% may be removed in a future release.

%   RSWEEP =
%   helperFMCWSimulate(NSWEEP,HWAVE,HRADARPLATFORM,HCARPLATFORM,HTX,
%   HCHANNEL,HCAR,HRX) returns the simulated sweep train RSWEEP. 
%
%   The input parameters are:
%       NSWEEP:             number of sweeps
%       HWAVE:              waveform object
%       HRADARPLATFORM:     platform object for the radar
%       HCARPLATFORM:       platform object for target car
%       HTX:                transmitter object
%       HCHANNEL:           propagation channel object
%       HCAR:               target car object
%       HRX:                receiver object
%
%   The rows of RSWEEP represent fast time and its columns represent slow
%   time (pulses). When the pulse transmitter uses staggered PRFs, the
%   length of the fast time sequences is determined by the highest PRF.

%   Copyright 2010-2014 The MathWorks, Inc.

release(hwav);
release(hradarplatform);
release(htx);
release(hrx);
num_targets = numel(hcar);
if num_targets > 1
    for m = 1:num_targets
        release(hcarplatform{m});
        release(hchannel{m});
        release(hcar{m});
    end
else
    release(hcarplatform);
    release(hchannel);
    release(hcar);    
end

if isa(hwav,'phased.MFSKWaveform')
    sweeptime = hwav.StepTime*hwav.StepsPerSweep;
else
    sweeptime = hwav.SweepTime;
end
Nsamp = round(hwav.SampleRate*sweeptime);

xr = complex(zeros(Nsamp,Nsweep));
xr_unmixed = xr;

for m = 1:Nsweep
    [radar_pos,radar_vel] = step(...
        hradarplatform,sweeptime);       % radar moves during sweep
    x = step(hwav);                           % generate the FMCW signal
    xt = step(htx,x);                         % transmit the signal
    if num_targets > 1
        xrtemp = complex(zeros(Nsamp,1));
        for n = 1:num_targets
            [tgt_pos,tgt_vel] = step(hcarplatform{n},... 
                sweeptime);                  % car moves during sweep
            xttemp = step(hchannel{n},xt,radar_pos,tgt_pos,...
                radar_vel,tgt_vel);               % propagate the signal
            xrtemp = xrtemp+step(hcar{n},xttemp); % reflect the signal
        end
    else
        [tgt_pos,tgt_vel] = step(hcarplatform,... 
            sweeptime);                  % car moves during sweep
        xttemp = step(hchannel,xt,radar_pos,tgt_pos,...
            radar_vel,tgt_vel);               % propagate the signal
        xrtemp = step(hcar,xttemp);           % reflect the signal
    end
    xrtemp = step(hrx,xrtemp);                % receive the signal
    xd = dechirp(xrtemp,x);                   % dechirp the signal
    xr_unmixed(:,m) = xrtemp;
    xr(:,m) = xd;                             % buffer the dechirped signal
end
