classdef (Sealed) CFARDetector < matlab.System & matlab.system.mixin.CustomIcon & ...
     matlab.system.mixin.Propagates
%CFARDetector   Constant false alarm rate (CFAR) detector
%   H = phased.CFARDetector creates a constant false alarm rate (CFAR)
%   detector System object, H. This object performs CFAR detection on the
%   input data.
%
%   H = phased.CFARDetector(Name,Value) creates a CFAR detector object, H,
%   with the specified property Name set to the specified Value. You can
%   specify additional name-value pair arguments in any order as
%   (Name1,Value1,...,NameN,ValueN).
%
%   Step method syntax:
%
%   Y = step(H,X,IDX) performs the CFAR detection on the real input data X.
%   X can be either a column vector or a matrix. Each row of X is a cell
%   and each column of X is independent data. Detection is performed along
%   each column for the cells specified in IDX. IDX must be a vector of
%   positive integers with each entry specifying the index of a cell under
%   test (CUT).
%
%   Y is an MxN matrix containing the logical detection result for the
%   cells in X. M is the number of indices specified in IDX and N is the
%   number of independent signals in X.
%
%   Y = step(H,X,IDX,K) uses K as the threshold factor used to calculate
%   the detection threshold when you set the ThresholdFactor property to
%   'Input port'. K must be a positive scalar.
%
%   [Y,TH] = step(H,X,IDX) returns additional output TH as the detection
%   threshold for each cell of interest in X when you set the
%   ThresholdOutputPort property to true. TH has the same dimensionality as
%   Y.
%
%   You can combine optional input and output arguments when their enabling
%   properties are set. Optional inputs and outputs must be listed in the
%   same order as the order of the enabling properties. For example,
%
%   [Y,TH] = step(H,X,IDX,K)
%
%   The algorithm used in CFARDetector is cell averaging CFAR. Detection is
%   performed in three steps. First, the training cells are identified from
%   the input and the values in these training cells are averaged to form
%   the noise estimate. Second, the noise estimate is multiplied by the
%   threshold factor to form the threshold. Finally, the value in the test
%   cell is compared against the threshold to determine whether the target
%   is present or absent. If the value is greater than the threshold, the
%   target is present.
%
%   CFARDetector methods:
%
%   step     - Perform CFAR detection (see above)
%   release  - Allow property value and input characteristics changes
%   clone    - Create CFAR detector object with same property values
%   isLocked - Locked status (logical)
%
%   CFARDetector properties:
%
%   Method                - CFAR algorithm
%   Rank                  - Rank of order statistic
%   NumGuardCells         - Number of guard cells
%   NumTrainingCells      - Number of training cells
%   ThresholdFactor       - Threshold factor method
%   ProbabilityFalseAlarm - Probability of false alarm
%   CustomThresholdFactor - Custom threshold factor
%   ThresholdOutputPort   - Output detection threshold
%
%   % Example:
%   %   Perform cell averaging CFAR detection on a given Gaussian noise
%   %   vector with a desired probability of false alarm of 0.1. Assume
%   %   that the data is from a square law detector and no pulse
%   %   integration is performed. Use 50 cells to estimate the noise
%   %   level and 1 cell to separate the test cell and training cells.
%   %   Perform the detection on all cells of input.
%
%   rs = RandStream.create('mt19937ar','Seed',5);
%   hdet = phased.CFARDetector('NumTrainingCells',50,'NumGuardCells',2,...
%                   'ProbabilityFalseAlarm',0.1);
%   N = 1000; x = 1/sqrt(2)*(randn(rs,N,1)+1i*randn(rs,N,1));
%   dresult = step(hdet,abs(x).^2,1:N);
%   Pfa = sum(dresult)/N
%
%   See also phased, phased.MatchedFilter, phased.TimeVaryingGain,
%   npwgnthresh.

%   Copyright 2009-2014 The MathWorks, Inc.

%   Reference
%   [1] Mark Richards, Fundamentals of Radar Signal Processing, 2005


%#ok<*EMCLS>
%#ok<*EMCA>
%#codegen

    properties (Nontunable)
        %Method     CFAR algorithm
        %   Specify the algorithm of CFAR detector using one of 'CA' |
        %   'SOCA' | 'GOCA' | 'OS', where the default is 'CA'. When you set
        %   the Method property to 'CA', the CFAR detector uses the
        %   cell-averaging algorithm. When you set the Method property to
        %   'SOCA', the CFAR detector uses the smallest-of cell-averaging
        %   algorithm. When you set the Method property to 'GOCA', the CFAR
        %   detector uses the greater-of cell-averaging algorithm. When you
        %   set the Method property to 'OS', the CFAR detector uses the
        %   order statistic algorithm.
        Method = 'CA'
        %Rank   Rank of order statistic
        %   Specify the rank of order statistic used in the order statistic
        %   CFAR algorithm as a positive integer. The value of the Rank
        %   property must be between 1 and N where N is the total number of
        %   training cells. The default value of this property is 1. This
        %   property only applies when you set the Method property to 'OS'.
        Rank = 1
        %NumGuardCells   Number of guard cells
        %   Specify the number of guard cells used in training as an even
        %   integer. It specifies the total number of cells on both sides
        %   of the cell under test. The default value of this property is 2
        %   indicating that there is one guard cell at both the front and
        %   back of the cell under test.
        NumGuardCells = 2
        %NumTrainingCells   Number of training cells
        %   Specify the number of training cells used in training as an
        %   even integer. Whenever possible, the training cells are equally
        %   divided before and after the cell under test. The default value
        %   of this property is 2 indicating that there is one training
        %   cell at both the front and back of the cell under test.
        NumTrainingCells = 2
        %ThresholdFactor    Threshold factor method
        %   Specify the method of obtaining the threshold factor using one
        %   of 'Auto' | 'Input port' | 'Custom', where the default is
        %   'Auto'. When you set the ThresholdFactor property to 'Auto',
        %   the threshold factor is calculated based on the desired
        %   probability of false alarm specified in the
        %   ProbabilityFalseAlarm property. The calculation assumes that
        %   each independent signal in the input is a single pulse coming
        %   out of a square law detector with no pulse integration. In
        %   addition, the noise is assumed to be white Gaussian. When you
        %   set the ThresholdFactor property to 'Input port', the threshold
        %   factor is specified through input argument. When you set the
        %   ThresholdFactor property to 'Custom', the threshold factor is
        %   the value of the CustomThresholdFactor property.
        ThresholdFactor = 'Auto'
        %ProbabilityFalseAlarm  Probability of false alarm
        %   Specify the desired probability of false alarm as a scalar
        %   between 0 and 1 (not inclusive). This property only applies
        %   when you set the ThresholdFactor property to 'Auto'. The
        %   default value of this property is 0.1.
        ProbabilityFalseAlarm = 0.1
    end

    properties
        %CustomThresholdFactor  Custom threshold factor
        %   Specify the custom threshold factor as a positive scalar. This
        %   property only applies when you set the ThresholdFactor property
        %   to 'Custom'. This property is tunable. The default value of
        %   this property is 1.
        CustomThresholdFactor = 1
    end 

    properties (Nontunable, Logical) 
        %ThresholdOutputPort    Output detection threshold
        %   Set this property to true to output the detection threshold.
        %   Set this property to false to not output the detection
        %   threshold. The default value of this property is false.
        ThresholdOutputPort = false;
    end
    
    properties (Access=private, Nontunable)
        cTraining;
        pMaximumCellIndex;
        pNumChannels;
    end
    
    properties (Access=private)
        pFactor;
    end
    
    properties(Constant, Hidden)
        ThresholdFactorSet = matlab.system.StringSet(...
            {'Auto','Input port','Custom'});
        MethodSet = matlab.system.StringSet(...
            {'CA','GOCA','SOCA','OS'});
    end

    methods
        
        function set.NumGuardCells(obj,val)
            validateattributes( val, { 'double' }, { 'finite', 'nonnan', 'nonnegative', 'even', 'scalar' }, '', 'NumGuardCells');
            obj.NumGuardCells = val;
        end
        
        function set.NumTrainingCells(obj,val)
            validateattributes( val, { 'double' }, { 'finite', 'nonnan', 'positive', 'even', 'scalar' }, '', 'NumTrainingCells');
            obj.NumTrainingCells = val;
        end
        
        function set.ProbabilityFalseAlarm(obj,val)
            % Pfa cannot be either 0 or 1 which makes Pd also 0 and 1.
            sigdatatypes.validateProbability(val,'phased.CFARDetector',...
                'ProbabilityFalseAlarm',{'scalar','>',0,'<',1});
            obj.ProbabilityFalseAlarm = val;
        end
        
        function set.CustomThresholdFactor(obj,val)
            validateattributes( val, { 'double' }, { 'nonempty', 'finite', 'positive', 'scalar' }, '', 'CustomThresholdFactor');
            obj.CustomThresholdFactor = val;
        end
        
        function set.Rank(obj,val)
            validateattributes(val, {'double'}, {'nonempty','finite','positive','scalar','integer'},'','Rank');
            obj.Rank = val;
        end
     end
    
    methods
        function obj = CFARDetector(varargin)
            setProperties(obj, nargin, varargin{:});
        end
    end
    
    methods (Access = protected)
        
        function validatePropertiesImpl(obj)
            if obj.Method(1) == 'O' % OS
                sigdatatypes.validateIndex(obj.Rank,...
                    '','Rank',{'<=',obj.NumTrainingCells});
            end
        end
        
        function num = getNumInputsImpl(obj) 
            if obj.ThresholdFactor(1) == 'I' %Input port
                num = 3;
            else
                num = 2;
            end
        end
        
        function num = getNumOutputsImpl(obj)
            if obj.ThresholdOutputPort
                num = 2;
            else
                num = 1;
            end
        end
        
        function releaseImpl(obj)
            release(obj.cTraining);
        end
        
        function resetImpl(obj)
            reset(obj.cTraining);
        end
        
        function flag = isInactivePropertyImpl(obj, prop)
            flag = false;
            if  (obj.ThresholdFactor(1) ~= 'A') && ... %Auto
                    strcmp(prop, 'ProbabilityFalseAlarm')
                flag = true;
            end
            if (obj.ThresholdFactor(1) ~= 'C') && ... %Custom
                    strcmp(prop, 'CustomThresholdFactor')
                flag = true;
            end
            if (obj.Method(1) ~='O') && ...  %OS
                    strcmp(prop, 'Rank')
                flag = true;
            end
        end
        
        function setupImpl(obj,X,CUTIdx,ThFac) %#ok<INUSD>
            sz_x = size(X);
            obj.pMaximumCellIndex = sz_x(1);
            obj.pNumChannels = sz_x(2);
           
            if (obj.Method(1) == 'S') || (obj.Method(1) == 'G') %SOCA || GOCA
                obj.cTraining = phased.internal.CFARTraining(...
                    'NumGuardCells',obj.NumGuardCells,...
                    'NumTrainingCells',obj.NumTrainingCells,...
                    'CombineTrainingData',false);
            else
                 obj.cTraining = phased.internal.CFARTraining(...
                    'NumGuardCells',obj.NumGuardCells,...
                    'NumTrainingCells',obj.NumTrainingCells,...
                    'CombineTrainingData',true);
            end
            if obj.ThresholdFactor(1) == 'A' %Auto
                obj.pFactor = calcWGNThresholdFactor(obj);
            elseif obj.ThresholdFactor(1) == 'C' %Custom
                processTunedPropertiesImpl(obj);
            end
        end
        
        function flag = isInputComplexityLockedImpl(obj,index) 
            flag = true;
            if (obj.ThresholdFactor(1) == 'I') && (index == 3)
                flag = true;
            end
        end
        
        function flag = isOutputComplexityLockedImpl(obj,~)  %#ok<INUSD>
            flag = true;
        end
        
        function validateInputsImpl(obj,x,cutidx,thfac)
            cond = ~isa(x,'double');
            if cond
                coder.internal.errorIf(cond, ...
                     'MATLAB:system:invalidInputDataType','X','double');
            end
            
            cond = ~ismatrix(x) || isempty(x);
            if cond
                coder.internal.errorIf(cond, ...
                     'MATLAB:system:inputMustBeMatrix','X');
            end
            
            cond = ~isreal(x);
            if cond
                coder.internal.errorIf(cond, ...
                     'phased:CFARDetector:ComplexInput', 'X');
            end
            
            cond = ~isa(cutidx,'double');
            if cond
                coder.internal.errorIf(cond, ...
                     'MATLAB:system:invalidInputDataType','Idx','double');
            end
            
            cond = ~isvector(cutidx) || isempty(cutidx);
            if cond
                coder.internal.errorIf(cond, ...
                     'MATLAB:system:inputMustBeVector','Idx');
            end
                        
            if obj.ThresholdFactor(1) == 'I'
                cond = ~isa(thfac,'double');
                if cond
                    coder.internal.errorIf(cond, ...
                         'MATLAB:system:invalidInputDataType','K','double');
                end
                
                cond = ~isscalar(thfac);
                if cond
                    coder.internal.errorIf(cond, ...
                         'MATLAB:system:inputMustBeScalar','K');
                end
                
                cond = ~isreal(thfac);
                if cond
                    coder.internal.errorIf(cond, ...
                         'phased:CFARDetector:ComplexInput', 'K');
                end
                
            end                
        end
        
        function processTunedPropertiesImpl(obj)
            if obj.ThresholdFactor(1) == 'C'
                obj.pFactor = obj.CustomThresholdFactor;
            end
        end
        
        function s = saveObjectImpl(obj)
            s = saveObjectImpl@matlab.System(obj);
            s.isLocked = isLocked(obj);
            if isLocked(obj)
                s.cTraining = saveobj(obj.cTraining);
                s.pFactor = obj.pFactor;
                s.pMaximumCellIndex = obj.pMaximumCellIndex;
                s.pNumChannels = obj.pNumChannels;
            end
        end
        
        function s = loadSubObjects(obj,s)
            if isfield(s,'isLocked')
                if s.isLocked
                    obj.cTraining = phased.internal.CFARTraining.loadobj(s.cTraining);
                    s = rmfield(s,'cTraining');
                end
                s = rmfield(s,'isLocked');
            end
        end
        
        function loadObjectImpl(obj,s,~)
            s = loadSubObjects(obj,s);
            fn = fieldnames(s);
            for m = 1:numel(fn)
                obj.(fn{m}) = s.(fn{m});
            end
        end
        
        function flag = isInputSizeLockedImpl(~,~)
            flag = true;
        end
        
        function [y,th] = stepImpl(obj,X,CUTIdx,ThFac)
        
            sigdatatypes.validateIndex(CUTIdx,'step','Idx',...
                {'<=',obj.pMaximumCellIndex});
            if obj.ThresholdFactor(1) ~= 'I'
                ThFac = obj.pFactor;
            else
                validateattributes(ThFac,{'double'},{'positive'},...
                    'step','K');
            end
            
            NumCUT = numel(CUTIdx);
            th = zeros(NumCUT,obj.pNumChannels);
            for m = 1:NumCUT
                if obj.Method(1) == 'C' %'CA'
                    trndata = step(obj.cTraining,X,CUTIdx(m));
                    % Averaging cells
                    noisepowerest = mean(trndata,1);
                elseif obj.Method(1) == 'S' %'SOCA'
                    [trndatal,trndatat] = step(obj.cTraining,X,CUTIdx(m));
                    noisepowerest = min(mean(trndatal,1),mean(trndatat,1));
                elseif obj.Method(1) == 'G' %'GOCA'
                    [trndatal,trndatat] = step(obj.cTraining,X,CUTIdx(m));
                    noisepowerest = max(mean(trndatal,1),mean(trndatat,1));
                else %'OS'
                    trndata = step(obj.cTraining,X,CUTIdx(m));
                    temp = sort(trndata,1,'ascend');
                    noisepowerest = temp(obj.Rank,:);
                end
                % Form threshold
                th(m,:) = noisepowerest * ThFac;
            end
            
            y = X(CUTIdx,:) > th;
        end
        
    end

    methods (Access = private)
        function alpha = calcWGNThresholdFactor(obj)
        %calcWGNThresholdFactor calculate threshold factor for white
        %                       Gaussian noise
        
            % currently we can only calculate threshold when the input is
            % single pulses, with no pulse integration performed.
            % single pulse threshold factor, see [1]
            Nc = obj.NumTrainingCells;
            Pfa = obj.ProbabilityFalseAlarm;
            CAThreshold = Nc*(Pfa^(-1/Nc)-1); 
            if obj.Method(1) == 'C'
                alpha = CAThreshold; 
            elseif obj.Method(1) == 'S'
                %alpha = fzero(@(x) SOCAWGNThresholdFactor(x,Nc,Pfa), CAThreshold);
                SOCAWGNThresholdFactor(Nc,Pfa);
                alpha = fzero(@SOCAWGNThresholdFactor,CAThreshold);
            elseif obj.Method(1) == 'G'
                %alpha = fzero(@(x) GOCAWGNThresholdFactor(x,Nc,Pfa), CAThreshold);
                GOCAWGNThresholdFactor(Nc,Pfa);
                alpha = fzero(@GOCAWGNThresholdFactor,CAThreshold);
            elseif obj.Method(1) == 'O'
                OSWGNThresholdFactor(Nc,obj.Rank,Pfa);
                
                if ~isempty(coder.target) %Codegen
                    alpha = fzero(@OSWGNThresholdFactor,CAThreshold);
                else % In MATLAB
                
                   % Try to find a threshold with initial guess. If fzero does 
                   % not converge, fzero throws an error. We configure fzero to 
                   % avoid throwing the error and increase the guess by 10 times 
                   % and try another call to fzero until max iteration reached.
                   % If fzero can still not find a solution, we throw a meaningful 
                   % error.
                
                   flag = -3;
                   initguess = 0.1*CAThreshold;
                   foption = optimset('Display','off');
                   maxiter = 50;
                   iter = 1;
                   while flag == -3 && iter <= maxiter
                       initguess = 10*initguess;
                       [alpha,~,flag] = fzero(@OSWGNThresholdFactor,...
                                              initguess,foption);
                       iter = iter+1;
                   end
                   if flag == -3
                       error(message('phased:CFARDetector:MaxIterForOS','Rank'));
                   end
                end
            end
        end
    end
    
    methods (Static,Hidden,Access=protected)  
        function header = getHeaderImpl
            header = matlab.system.display.Header(...
                'Title',getString(message('phased:library:block:CFARDetectorTitle')),...
                'Text',getString(message('phased:library:block:CFARDetectorDesc',...
                'CA','GOCA','SOCA','OS')));
        end
    end

    methods (Access = protected)
        function varargout = getInputNamesImpl(~)
            varargout = {'X','Idx','K'};
        end
        
        function varargout = getOutputNamesImpl(~)
            varargout = {'Y','Th'};
        end
               
        function str = getIconImpl(obj) 
            str = sprintf('%s CFAR',obj.Method);
        end
        function varargout = getOutputSizeImpl(obj)
            szX = propagatedInputSize(obj,1);
            szCut = propagatedInputSize(obj,2);
            varargout{1} = [max(szCut) szX(2)];
            varargout{2} = varargout{1};
        end
        function varargout = isOutputFixedSizeImpl(obj)
        %Fixed if both X and CUTIDX are fixed.
            varargout{1} = propagatedInputFixedSize(obj, 1) && ...
                propagatedInputFixedSize(obj, 2);
            varargout{2} = varargout{1};            
        end
        function varargout = getOutputDataTypeImpl(obj)
            varargout{1} = 'logical';
            varargout{2} = propagatedInputDataType(obj,1);
        end
        function varargout = isOutputComplexImpl(~)
            varargout = {false, false};
        end        
    end
end

function c = GOCASOCAThresholdCore(x,N)

temp = 0;
for k = 0:N/2-1
    tempval = gammaln(N/2+k)-gammaln(k+1)-gammaln(N/2);
    temp = temp+exp(tempval)*(2+x/(N/2))^(-k);
end
c = temp*(2+x/(N/2))^(-N/2);
    
end

function y = SOCAWGNThresholdFactor(varargin)

persistent N pfa;
if nargin > 1
    N = varargin{1};
    pfa = varargin{2};
else
    % Lines in isempty conditionals are dead code
    % The isempty conditionals were added since codegen
    % was complaining that those variables were undefined
    % in certain code paths. Compiler could not detect
    % that the persistent variables were always set first
    % by calling the function with two arguments. The latter
    % workaround was added since codegen did not support
    % anonymous functions.
    
    if isempty(N)
     N = 0;
    end    
    if isempty(pfa)
     pfa = 0;
    end
    
    % [1] (7.35)
    y = GOCASOCAThresholdCore(varargin{1},N)-pfa/2;
end
end

function y = GOCAWGNThresholdFactor(varargin)

persistent N pfa;
if nargin > 1
    N = varargin{1};
    pfa = varargin{2};
else
    if isempty(N)
        N = 0;
    end
    if isempty(pfa)
        pfa = 0;
    end
    
    x = varargin{1};
    % [1] (7.37)
    y = (1+x/(N/2))^(-N/2)-GOCASOCAThresholdCore(x,N)-pfa/2;
end
end

function y = OSWGNThresholdFactor(varargin)
persistent N k pfa;
if nargin > 1
    N = varargin{1};
    k = varargin{2};
    pfa = varargin{3};
else
    if isempty(N)
        N = 0;
    end
    if isempty(pfa)
        pfa = 0;
    end
    if isempty(k)
        k = 0;
    end
    x = varargin{1};
    % [1] (7.49)
    temp1 = gammaln(N+1)-gammaln(k)-gammaln(N-k+1);
    c = x+N-k+1;
    if c >= 0
        temp2 = betaln(c,k);
        y = exp(temp1+temp2)-pfa;
    else % betaln used to return inf, now error out. Restore the behavior
        y = nan;    
    end
end
end
