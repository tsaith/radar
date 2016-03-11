classdef (Sealed) FreeSpace < phased.internal.AbstractSampleRateEngine & ...
        matlab.system.mixin.Propagates & matlab.system.mixin.CustomIcon & ...
        matlab.system.mixin.internal.SampleTime
%FreeSpace Free space environment
%   H = phased.FreeSpace creates a free space environment System object, H.
%   This object simulates narrowband signal propagation in free space, by
%   applying range-dependent time delay, gain and phase shift to the input
%   signal.
%
%   H = phased.FreeSpace(Name,Value) returns a free space environment
%   object, H, with the specified property Name set to the specified
%   Value. You can specify additional name-value pair arguments in any
%   order as (Name1,Value1,...,NameN,ValueN).
%
%   Step method syntax:
%
%   Y = step(H,X,POS1,POS2,VEL1,VEL2) returns the resulting signal Y when
%   the narrowband signal X propagates in free space from either the
%   position of a source POS1 to the positions of one or more destinations
%   POS2 or from the positions of several sources to the position of one
%   destination. VEL1 and VEL2 specify the velocities of the sources and
%   destinations respectively. POS1 and POS2 have a size of either 3x1 and
%   3xN respectively or 3xN and 3x1 where each column is in the form of [x;
%   y; z] (in meters). Similarly VEL1 and VEL2 have a size of either 3x1
%   and 3xN respectively or 3xN and 3x1 where each column is in the the
%   form of [Vx; Vy; Vz] (in meters/second). N is the number of signals to
%   propagate and can only be equal to 1 when the signal is polarized.
%
%   X can be either an N column matrix or a struct. If X is a matrix, Y is
%   a matrix with same dimensions and each column of X and Y represent the
%   signal at the source and destination respectively of a propagation
%   path. The propagation paths are defined in the order of the positions
%   specified in POS1 and POS2. If X is a struct, it must contain either X,
%   Y, and Z fields or H and V fields. The X, Y, and Z fields represent the
%   x, y, and z components of the polarized signal, respectively and the H
%   and V fields represent the horizontal and vertical components of the
%   polarized signal, respectively. In this case, the output Y is also a
%   struct containing the same fields as X. Each field in Y contains the
%   resulting signal of the corresponding field in X,
%
%   The output Y represents the signals arriving at the propagation
%   destinations within the current time frame, which is the time occupied
%   by the current input. If it takes longer than the current time frame
%   for the signals to propagate from the origin to the destination, then
%   the output contains no contribution from the input of the current time
%   frame. The output Y can be written as
%
%   Y(t) = X(t-tau)/L
%
%   where tau is the delay and L is the propagation loss. The delay tau can
%   be calculated as R/c where R is the propagation distance and c is the
%   propagation speed. The free space path loss is given by
%
%   L = (4*pi*R/lambda)^2
%
%   where lambda is the signal wavelength.
%
%   FreeSpace methods:
%
%   step     - Propagate signal from one location to another (see above)
%   release  - Allow property value and input characteristics changes
%   clone    - Create free space object with same property values
%   isLocked - Locked status (logical)
%   reset    - Reset internal states of the propagation channel
%
%   FreeSpace properties:
%
%   PropagationSpeed      - Propagation speed 
%   OperatingFrequency    - Signal carrier frequency 
%   TwoWayPropagation     - Perform two-way propagation
%   SampleRate            - Sample rate 
%   MaximumDistanceSource - Source of maximum one-way propagation distance
%   MaximumDistance       - Maximum one-way propagation distance
%
%   % Example:
%   %   Calculate the result of propagating a signal in a free space 
%   %   environment from a radar at (1000, 0, 0) to a target at (300, 200,
%   %   50). Assume both the radar and the target are stationary.
%
%   henv = phased.FreeSpace('SampleRate',8e3);
%   y = step(henv,ones(10,1),[1000; 0; 0],[300; 200; 50],[0;0;0],[0;0;0])
%
%   See also phased, phased.RadarTarget, fspl.

%   Copyright 2010-2014 The MathWorks, Inc.

%   Reference
%   [1] John Proakis, Digital Communications, 4th Ed., McGraw-Hill, 2001
%   [2] Merrill Skolnik, Introduction to Radar Systems, 3rd Ed.,
%       McGraw-Hill, 2001 
%   [3] Mark Richards, Fundamentals of Radar Signal Processing,
%       McGraw-Hill, 2005


%#ok<*EMCLS>
%#ok<*EMCA>
%#codegen

    properties (Nontunable)
        %PropagationSpeed Propagation speed (m/s)
        %   Specify the wave propagation speed (in m/s) in free space as a
        %   scalar. The default value of this property is the speed of
        %   light.
        PropagationSpeed = physconst('LightSpeed')
        %OperatingFrequency Signal carrier frequency (Hz)
        %   Specify the carrier frequency (in Hz) of the narrowband signal
        %   as a scalar. The default value of this property is 3e8 (300
        %   MHz).
        OperatingFrequency = 3e8       
    end

    properties (Nontunable, Logical) 
        %TwoWayPropagation Perform two-way propagation
        %   Set this property to true to perform two-way propagation. Set
        %   this property to false to perform one-way propagation. The
        %   default value of this property is false.
        TwoWayPropagation = false
    end
    
    properties (Nontunable)
        %SampleRate Sample rate (Hz)
        %   Specify the sample rate (in Hz) as a scalar. The default value
        %   of this property is 1e6 (1 MHz).
        SampleRate = 1e6    
    end
    
    properties (Nontunable)
        %MaximumDistanceSource  Source of maximum one-way propagation 
        %                       distance
        %   Specify how the maximum one-way propagation distance is
        %   specified as one of 'Auto' | 'Property', where the default is
        %   'Auto'. When you set this property to 'Auto', FreeSpace
        %   automatically allocates the memory to simulate the propagation
        %   delay. When you set this property to 'Property', the maximum
        %   one-way propagation distance is specified via MaximumDistance
        %   property and any signal that needs to propagation more than
        %   MaximumDistance one way is ignored. 
        %
        %   To use FreeSpace in MATLAB Function Block in Simulink, set this
        %   property to 'Property'.
        MaximumDistanceSource = 'Auto'
        %MaximumDistance    Maximum one-way propagation distance (m)
        %   Specify the maximum one-way propagation distance (in meters) as
        %   a positive scalar. This property applies when you set the
        %   MaximumDistanceSource property to 'Property'. The default value
        %   of this property is 10e3.
        MaximumDistance = 10e3
    end

    properties(Constant, Hidden)
        MaximumDistanceSourceSet = dsp.CommonSets.getSet('AutoOrProperty');
    end
    
    properties (Access = private, Nontunable)
        %Internal buffer
        cBuffer
        %Wavelength
        pLambda
        %Propagation range factor, one-way or round trip
        pRangeFactor
        %Valid field names
        pValidFields 
        %Sample rate, in MATLAB, specified by property but in Simulink,
        %specified by engine
        pSampleRate
    end
    
    properties (Access = private, Logical, Nontunable)
        %Whether input is a struct
        pIsInputStruct
         %Codegen mode
        pIsCodeGen = false
    end
    
    methods
        function set.OperatingFrequency(obj,value)
            validateattributes(value,{'double'},{'scalar','finite',...
                'positive'},'FreeSpace','OperatingFrequency');
            obj.OperatingFrequency = value;
        end
        function set.SampleRate(obj,value)
            validateattributes(value,{'double'},{'scalar','positive',...
                'finite'},'FreeSpace','SampleRate');
            obj.SampleRate = value;
        end
        function set.PropagationSpeed(obj,value)
            sigdatatypes.validateSpeed(value,'FreeSpace',...
                'PropagationSpeed',{'scalar','positive'});
            obj.PropagationSpeed = value;
        end
        function set.MaximumDistance(obj,value)
            sigdatatypes.validateDistance(value,'FreeSpace',...
                'MaximumDistance',{'scalar','positive'});
            obj.MaximumDistance = value;
        end
    end

    methods
        function obj = FreeSpace(varargin)
            setProperties(obj, nargin, varargin{:});
            if isempty(coder.target())
                obj.pIsCodeGen = false;
            else
                obj.pIsCodeGen = true;
            end
        end
    end

    methods (Access = protected)

        function flag = isInactivePropertyImpl(obj, prop)
            if (obj.MaximumDistanceSource(1) == 'A') && ...  %Auto
                    strcmp(prop, 'MaximumDistance')
                flag = true;
            else
                flag = false;
            end
        end
        
        function validateInputsImpl(~,x,startLoc,endLoc,baseVel,targetVel)
            coder.extrinsic('mat2str');
            coder.extrinsic('num2str');            
            if isstruct(x)
                flag_hasXYZ = isfield(x(1),'X') && isfield(x(1),'Y') && isfield(x(1),'Z');
                flag_hasHV = isfield(x(1),'H') && isfield(x(1),'V');
                cond = ~flag_hasXYZ && ~flag_hasHV;
                if cond
                    coder.internal.errorIf(cond,'phased:polarization:invalidPolarizationStruct');
                end
                cond = ~isscalar(x);
                if cond
                    coder.internal.errorIf(cond,'phased:polarization:invalidPolarizationArrayStruct','X');
                end
                if flag_hasXYZ
                    x_x = x.X;
                    x_y = x.Y;
                    x_z = x.Z;
                    cond =  ~isa(x_x,'double');
                    if cond
                        coder.internal.errorIf(cond, ...
                             'MATLAB:system:invalidInputDataType','X.X','double');
                    end
                    cond =  ~iscolumn(x_x) || isempty(x_x);
                    if cond
                        coder.internal.errorIf(cond, ...
                             'MATLAB:system:inputMustBeColVector','X.X');
                    end
                    cond =  ~isa(x_y,'double');
                    if cond
                        coder.internal.errorIf(cond, ...
                             'MATLAB:system:invalidInputDataType','X.Y','double');
                    end
                    cond =  ~iscolumn(x_y) || isempty(x_y);
                    if cond
                        coder.internal.errorIf(cond, ...
                             'MATLAB:system:inputMustBeColVector','X.Y');
                    end
                    cond =  ~isa(x_z,'double');
                    if cond
                        coder.internal.errorIf(cond, ...
                             'MATLAB:system:invalidInputDataType','X.Z','double');
                    end
                    cond =  ~iscolumn(x_z) || isempty(x_z);
                    if cond
                        coder.internal.errorIf(cond, ...
                             'MATLAB:system:inputMustBeColVector','X.Z');
                    end
                    cond = numel(x_x)~=numel(x_y) || numel(x_x)~=numel(x_z);
                    if cond
                        coder.internal.errorIf(cond,'phased:polarization:polarizationStructDimensionMismatch',...
                            'X,Y,Z','X');
                    end
                end
                if flag_hasHV
                    x_h = x.H;
                    x_v = x.V;
                    cond =  ~isa(x_h,'double');
                    if cond
                        coder.internal.errorIf(cond, ...
                             'MATLAB:system:invalidInputDataType','X.H','double');
                    end
                    cond =  ~iscolumn(x_h) || isempty(x_h);
                    if cond
                        coder.internal.errorIf(cond, ...
                             'MATLAB:system:inputMustBeColVector','X.H');
                    end
                    cond =  ~isa(x_v,'double');
                    if cond
                        coder.internal.errorIf(cond, ...
                             'MATLAB:system:invalidInputDataType','X.V','double');
                    end
                    cond =  ~iscolumn(x_v) || isempty(x_v);
                    if cond
                        coder.internal.errorIf(cond, ...
                             'MATLAB:system:inputMustBeColVector','X.V');
                    end
                    cond = numel(x_h)~=numel(x_v) ;
                    if cond
                        coder.internal.errorIf(cond,'phased:polarization:polarizationStructDimensionMismatch',...
                            'H,V','X');
                    end
                end
                if flag_hasXYZ && flag_hasHV
                    cond = numel(x_x)~=numel(x_h) ;
                    if cond
                        coder.internal.errorIf(cond,'phased:polarization:polarizationStructDimensionMismatch',...
                            'X,Y,Z,H,V','X');
                    end
                end
                numOfPropPaths = 1;  
            else
                cond =  ~isa(x,'double');
                if cond
                    coder.internal.errorIf(cond, ...
                         'MATLAB:system:invalidInputDataType','X','double');
                end
                numOfPropPaths = size(x,2);  
            end
            cond =  ~isa(startLoc,'double');
            if cond
                coder.internal.errorIf(cond, ...
                     'MATLAB:system:invalidInputDataType','Pos1','double');
            end
            if numOfPropPaths == 1
                expDim = '[3 ';
            else
                expDim = '[3 1] or [3 ';
            end
            startLocSize = size(startLoc);
            cond =  ~(isequal(startLocSize,[3 numOfPropPaths]) || isequal(startLocSize,[3 1]));
            if cond
                coder.internal.errorIf(cond, ...
                     'MATLAB:system:invalidInputDimensions','Pos1',...
                        [expDim coder.const(num2str(numOfPropPaths)) ']'], ...
                               coder.const(mat2str(startLocSize)));
            end
            cond =  ~isreal(startLoc);
            if cond
                coder.internal.errorIf(cond, ...
                     'phased:step:NeedReal', 'Pos1');
            end

            cond =  ~isa(endLoc,'double');
            if cond
                coder.internal.errorIf(cond, ...
                     'MATLAB:system:invalidInputDataType','Pos2','double');
            end
            endLocSize = size(endLoc);

            cond =  ~(isequal(endLocSize,[3 numOfPropPaths]) || isequal(endLocSize,[3 1]));
            if cond
                coder.internal.errorIf(cond, ...
                     'MATLAB:system:invalidInputDimensions','Pos2',...
                        [expDim coder.const(num2str(numOfPropPaths)) ']'], ...
                               coder.const(mat2str(endLocSize)));
            end
            cond =  ~isreal(endLoc);
            if cond
                coder.internal.errorIf(cond, ...
                     'phased:step:NeedReal', 'Pos2');
            end
            cond =   ~(isequal(startLocSize,[3 numOfPropPaths]) || ...
                      isequal(endLocSize,[3 numOfPropPaths]));
            if cond
                coder.internal.errorIf(cond, ...
                     'phased:phased:FreeSpace:AtLeastOneNotColumnVect','Pos1','Pos2',numOfPropPaths);
            end
            
            cond = ~ (isequal(startLocSize,[3 1]) || isequal(endLocSize,[3 1]));
            if cond
                coder.internal.errorIf(cond, ...
                     'phased:phased:FreeSpace:AtLeastOneColumnVect','Pos1','Pos2');
            end
            
            
            cond =   ~isa(baseVel,'double');
            if cond
                coder.internal.errorIf(cond, ...
                     'MATLAB:system:invalidInputDataType','Vel1','double');
            end
            baseVelSize = size(baseVel);
            cond =   ~isequal(baseVelSize,startLocSize);
            if cond
                coder.internal.errorIf(cond, ...
                     'MATLAB:system:invalidInputDimensions','Vel1',...
                     coder.const(mat2str(startLocSize)),coder.const(mat2str(baseVelSize)));
            end
            cond =   ~isreal(baseVel);
            if cond
                coder.internal.errorIf(cond, ...
                      'phased:step:NeedReal', 'Vel1');
            end

            cond =   ~isa(targetVel,'double');
            if cond
                coder.internal.errorIf(cond, ...
                      'MATLAB:system:invalidInputDataType','Vel2','double');
            end
            targetVelSize = size(targetVel);
            cond =   ~isequal(targetVelSize,endLocSize);
            if cond
                coder.internal.errorIf(cond, ...
                      'MATLAB:system:invalidInputDimensions','Vel2',...
                       coder.const(mat2str(endLocSize)),coder.const(mat2str(targetVelSize)));
            end
            cond =   ~isreal(targetVel);
            if cond
                coder.internal.errorIf(cond, ...
                     'phased:step:NeedReal', 'Vel2');
            end
        end

      function setupImpl(obj,x,~,~,~,~)
            obj.pIsInputStruct = isstruct(x);
            if obj.pIsInputStruct
                 flag_hasXYZ = isfield(x(1),'X') && isfield(x(1),'Y') && isfield(x(1),'Z');
                 flag_hasHV = isfield(x(1),'H') && isfield(x(1),'V');
                 if flag_hasXYZ
                    if flag_hasHV
                        obj.pValidFields = 'XYZHV';                                
                    else
                        obj.pValidFields = 'XYZ';        
                    end
                    sz_x = size(x.X,1);
                else
                    obj.pValidFields = 'HV';  
                    sz_x = size(x.H,1);
                end
            else
                sz_x = size(x,1);
            end
            obj.pSampleRate = getSampleRate(obj,sz_x,1,obj.SampleRate);
            if obj.TwoWayPropagation
                obj.pRangeFactor = 2;
            else
                obj.pRangeFactor = 1;
            end
            if strcmp(obj.MaximumDistanceSource,'Auto')
                obj.cBuffer = phased.internal.CircularBuffer(...
                    'BufferLength',1);
            else
                buflen = ceil(obj.pRangeFactor*...
                    obj.MaximumDistance/obj.PropagationSpeed*obj.pSampleRate);
                obj.cBuffer = phased.internal.CircularBuffer(...
                    'FixedLengthBuffer',true,'BufferLength',buflen);
            end
            obj.pLambda = obj.PropagationSpeed/obj.OperatingFrequency;
            
        end
        
        function flag = isInputComplexityLockedImpl(obj,index)  %#ok<INUSL>
            if index == 1
                flag = false;
            else % (index == 2,3,4,5)
                flag = true;
            end
        end
        
        function flag = isOutputComplexityLockedImpl(obj,~)  %#ok<INUSD>
            flag = false;
        end

        function releaseImpl(obj)
            release(obj.cBuffer);
        end

        function resetImpl(obj)
            reset(obj.cBuffer);
        end

        function s = saveObjectImpl(obj)
            s = saveObjectImpl@matlab.System(obj);
            if isLocked(obj)
                s.pLambda = obj.pLambda;
                s.pRangeFactor = obj.pRangeFactor;
                s.cBuffer = saveobj(obj.cBuffer);
                s.pIsInputStruct = obj.pIsInputStruct;
                s.pValidFields = obj.pValidFields;
                s.pSampleRate = obj.pSampleRate;
            end
        end

        function s = loadSubObjects(obj,s,wasLocked)
            if isfield(s,'isLocked')                                        
                if s.isLocked                                                   
                    obj.cBuffer = phased.internal.CircularBuffer.loadobj(s.cBuffer); 
                    s = rmfield(s,'cBuffer');
                    % recover locked sample rate information
                    obj.pSampleRate = s.SampleRate;
                end
                s = rmfield(s,'isLocked');                                      
            elseif wasLocked
                obj.cBuffer = phased.internal.CircularBuffer.loadobj(s.cBuffer);
                s = rmfield(s,'cBuffer');
                % recover locked sample rate information
                if isfield(s,'pSampleRate')
                    obj.pSampleRate = s.pSampleRate;
                    s = rmfield(s,'pSampleRate');
                else
                    obj.pSampleRate = s.SampleRate;
                end
            end
        end

        function loadObjectImpl(obj,s,wasLocked)
            s = loadSubObjects(obj,s,wasLocked);
            fn = fieldnames(s);
            for m = 1:numel(fn)
                obj.(fn{m}) = s.(fn{m});
            end
        end

        function flag = isInputSizeLockedImpl(~,~)
            flag = true;
        end

        function y = stepImpl(obj,x_in,startLoc,endLoc,baseVel,targetVel)

            k = obj.pRangeFactor;
            Fs = obj.pSampleRate;
            % Apply propagation loss and phase variation
            lambda = obj.pLambda;
            numOfStartLoc = size(startLoc,2);
            numOfEndLoc = size(endLoc,2);
            numOfPropPaths = size(x_in,2);
            propdistance = zeros(1,numOfPropPaths);
            rspeed = propdistance;
          % propagation distance
            for sIdx = 1:numOfStartLoc
                for eIdx = 1:numOfEndLoc
                    % propagation distance
                    propdistance(sIdx+eIdx-1) = norm(startLoc(:,sIdx) - endLoc(:,eIdx));
                    % add intra pulse Doppler
                    rspeed(sIdx+eIdx-1) = calcRadialSpeed(endLoc(:,eIdx),targetVel(:,eIdx),startLoc(:,sIdx),baseVel(:,sIdx));
                end
            end
            cond =  any(propdistance == 0);
            if cond
                coder.internal.errorIf(cond, ...
                     'phased:phased:FreeSpace:InvalidPropagationDistance');
            end
            sploss = k*fspl(propdistance,lambda); % spreading loss
            plossfactor = sqrt(db2pow(sploss));
            propdelay = k*propdistance/obj.PropagationSpeed;
            
            if obj.pIsInputStruct
                % can be moved to setup
                fn = obj.pValidFields;
                sz_field = size(x_in.(fn(1)));
                num_fn = numel(fn);
                x = complex(zeros([prod(sz_field) num_fn])); 
                for m = coder.unroll(1:numel(fn))
                    x(:,m) = x_in.(fn(m));
                end
            else
                x = x_in;
            end
            
            % only consider the time relative to the current pulse start
            % because all previous pulse time is absorbed into the phase
            % term given by the propagation distance as propagation
            % distance is updated by each pulse.
            
            if  numOfPropPaths ~= 1
                tempx = complex(zeros(size(x)));
                for pIdx = 1:numOfPropPaths
                    tempx(:,pIdx) = exp(-1i*2*pi*k*propdistance(pIdx)/lambda)/plossfactor(pIdx)*...
                            bsxfun(@times,x(:,pIdx),...
                                   exp(1i*2*pi*k*rspeed(pIdx)/lambda*(propdelay(pIdx)+(0:size(x,1)-1)'/Fs)));                    
                end
            else
                tempx = exp(-1i*2*pi*k*propdistance/lambda)/plossfactor*...
                        bsxfun(@times,x,...
                               exp(1i*2*pi*k*rspeed/lambda*(propdelay+(0:size(x,1)-1)'/Fs)));
            end
            % Calculate propagation delay in samples
            nDelay = propdelay*Fs;
            for pIdx = 1:numOfPropPaths
                if ~rem(propdelay(pIdx),1/Fs)
                    nDelay(pIdx) = round(nDelay(pIdx));
                else
                    nDelay(pIdx) = ceil(nDelay(pIdx));
                end
            end
            y_out = step(obj.cBuffer,tempx,nDelay);
            
            if obj.pIsInputStruct
                if ~obj.pIsCodeGen
                    y = x_in;
                end
                for m = coder.unroll(1:num_fn)
                    y.(fn(m)) = reshape(y_out(:,m),sz_field);
                end
            else
                y = y_out;
            end
            
        end

        function num = getNumInputsImpl(obj)   %#ok<MANU>
            num = 5;
        end
    end
    
    methods (Access = protected, Static, Hidden)
        function header = getHeaderImpl
          header = matlab.system.display.Header(...
              'Title',getString(message('phased:library:block:FreeSpaceTitle')),...
              'Text',getString(message('phased:library:block:FreeSpaceDesc')));
        end
        function groups = getPropertyGroupsImpl
            groups = matlab.system.display.Section(...
                'phased.FreeSpace');
            dMaximumDistanceSource = matlab.system.display.internal.Property(...
                'MaximumDistanceSource','IsGraphical',false,...
                'UseClassDefault',false,'Default','Property');
            dSampleRate = matlab.system.display.internal.Property(...
                'SampleRate','IsObjectDisplayOnly',true);
            for m = 1:numel(groups.PropertyList)
                if strcmp(groups.PropertyList{m},'MaximumDistanceSource')
                    groups.PropertyList{m} = dMaximumDistanceSource;
                elseif strcmp(groups.PropertyList{m},'SampleRate')
                    groups.PropertyList{m} = dSampleRate;
                end
            end
        end
    end
    
    methods (Access = protected)
        
        function sz_out = getOutputSizeImpl(obj)
            sz_out = inputSize(obj,1);
        end
        
        function dt_out = getOutputDataTypeImpl(obj)
            dt_out = inputDataType(obj,1);
        end
        
        function cp_out = isOutputComplexImpl(obj) %#ok<MANU>
            cp_out = true;
        end
        
        function fsz_out = isOutputFixedSizeImpl(obj) %#ok<MANU>
            fsz_out = true;
        end
        
        function varargout = getInputNamesImpl(obj)   %#ok<MANU>
            varargout = {'X','Pos1','Pos2','Vel1','Vel2'};
        end

        function varargout = getOutputNamesImpl(obj)  %#ok<MANU>
            varargout = {''};
        end
        
        function str = getIconImpl(obj) %#ok<MANU>
            str = sprintf('Free Space\nChannel');
        end
    end
    
end

function rspeed = calcRadialSpeed(tgtpos,tgtvel,refpos,refvel)
%calcRadialSpeed    Compute radial speed
%   RSPEED = calcRadialSpeed(POS,VEL,REFPOS,REFVEL) compute the relative
%   speed RSPEED (in m/s) for a target at position POS (in meters) with a
%   velocity VEL (in m/s) relative to the reference position REFPOS (in
%   meters) and reference velocity REFVEL (in m/s).

%   This is the same functionality as radialspeed function. However,
%   because we already done the input validation here, we want to skip the
%   validation to improve the performance. In addition, here all position
%   and velocity are always column vectors and the target and reference can
%   never be colocated, so it simplifies the computation too.

tgtdirec = tgtpos-refpos;
veldirec = tgtvel-refvel;

%Take the 2-norm of veldirec and tgtdirec
rn = sqrt(sum(tgtdirec.^2));

% negative sign to ensure that incoming relative speed is positive
rspeed = -(sum(veldirec.*tgtdirec)./rn);

end
