classdef rlLQTAgent < rl.agent.CustomAgent
    % rlLQTAgent: Creates a LQT agent.
    %
    %   agent = rlLQTAgent(CRITIC) creates a LQT agent with default
    %   options and the specified critic representation.
    %
    %   agent = rlLQTAgent(CRITIC,OPTIONS) creates a LQT agent with
    %   the specified options. To create OPTIONS, use rlLQTAgentOptions.
    %
    
    % ver1.0.0 2020-02-11 T.Iwata Test create
    
    % TODO
    %   Experience buffer������
    %   Init K ������
    %   w0�̏������@������
    
    %% Public Properties
    properties (Dependent)
        % Options to configure RL agent
        AgentOptions
    end
    
    properties (Access = private)
        % Private options to configure RL agent
        AgentOptions_ = [];
    end
    
    properties

        % Feedback gain
        K

        % Critic
        Critic
        
        % TD error buffer
        TDBuffer
        TDBufferSize = 1
        
        % Buffer for K
        KBuffer  
        % Number of updates for K
        KUpdate = 1

        
        
        % Stop learning value
        % �Q�C���̍X�V�������̒l�ȉ��ɂȂ�����w�K���I������
        stopLearningValue
        
    end
    
    properties (Access = private)
        % 1�C�e���[�V����������̃X�e�b�v���i���̐��ň�x����̍X�V���s���j
        StepNumPerIteration
        
        stepMode
        
        experienceBuffer
        experienceBufferCount = 0
    end
    
    
    %% MAIN METHODS
    methods
        % Constructor
        function this = rlLQTAgent(varargin)
            % input parser
            narginchk(2, 3);  % �����̐����m�F�i�ŏ�:2, �ő�:3�j
            % Call the abstract class constructor
            this = this@rl.agent.CustomAgent();
            
            % validate inputs
            % see also: rl.util.parseAgentInputs.m
            % infomation check
            oaInfo = varargin(cellfun(@(x) isa(x, 'rl.util.RLDataSpec'), varargin));
            if numel(oaInfo) ~= 2
                error('Action or obsevation infomation is invalid');
            end
            
            % options check
            UseDefault = false;
            opt = varargin(cellfun(@(x) isa(x, 'rl.option.AgentGeneric'), varargin));
            
            if numel(varargin)~=( numel(oaInfo)+numel(opt) )
                error(message('rl:agent:errInvalidAgentInput'));
            end
            
            if isempty(opt)
                opt{1} = rlLQTAgentOptions;
                UseDefault = true;
            else
                % check otption is compatible
                if ~isa(opt{1}, 'rl.option.rlLQTAgentOptions')
                    error(message('rl:agent:errMismatchedOption'));
                end
            end
            
            % set agent option
            this.AgentOptions = opt{1};
            
            % set ActionInfo and ObservationInfo
            this.ObservationInfo = oaInfo{1};
            this.ActionInfo = oaInfo{2};

            % Create the critic representation
            this.Critic = createCritic(this);

            % Initialize the gain matrix
%             this.K = rand(1, this.ObservationInfo.Dimension(1));
            this.K = [0.3 1.3 0.75];
%             
%             % Initialize learning parameters
%             this.stopLearningValue = p.Results.StopLearningValue;
% 
            % Initialize the log buffers
            this.KBuffer = cell(1,1000);
            this.KBuffer{1} = this.K;
        end
        
        function set.AgentOptions(this, NewOptions)
            validateattributes(NewOptions,{'rl.option.rlLQTAgentOptions'},{'scalar'},'','AgentOptions');
            
            this.AgentOptions_ = NewOptions;
            this.SampleTime = NewOptions.SampleTime;
            this.StepNumPerIteration = NewOptions.StepNumPerIteration;
        end
        function Options = get.AgentOptions(this)
            Options = this.AgentOptions_;
        end
    end
    
    %% Implementation of abstract parent protected methods
    methods (Access = protected)
        function action = getActionWithExplorationImpl(obj,Observation)
            % Given the current observation, select an action
            action = getAction(obj,Observation);
            
            % Add random noise to action
            action = action + 2*randn(size(action, 1),1);
        end
        % learn from current experiences, return action with exploration
        % exp = {state,action,reward,nextstate,isdone}
        function action = learnImpl(obj,exp)
            gamma = obj.AgentOptions.DiscountFactor;
            
            % Store in experience buffer
            obj.experienceBufferCount = obj.experienceBufferCount + 1;
            obj.experienceBuffer{obj.experienceBufferCount} = exp;
            
            
            
            % Wait N steps before updating critic parameters
            N = obj.StepNumPerIteration;
            
            if obj.experienceBufferCount>=N
                oaDim = obj.ObservationInfo.Dimension(1) + obj.ActionInfo.Dimension(1);
                yBuf = zeros(obj.experienceBufferCount,1);
                hBuf = zeros(obj.experienceBufferCount,0.5*oaDim*(oaDim+1));
                TDError = zeros(obj.experienceBufferCount, 1);
                for i = 1 : obj.experienceBufferCount
                    % Parse the experience input
                    x = obj.experienceBuffer{i}{1}{1};
                    u = obj.experienceBuffer{i}{2}{1};
                    r = obj.experienceBuffer{i}{3};
                    dx = obj.experienceBuffer{i}{4}{1};
                    
                    % In the linear case, critic evaluated at (x,u) is Q1 = theta'*h1,
                    % critic evaluated at (dx,-K*dx) is Q2 = theta'*h2. The target
                    % is to obtain theta such that Q1 - gamma*Q2 = y, that is,
                    % theta'*H = y. Following is the least square solution.
                    h1 = computeQuadraticBasis(x,u,oaDim);
                    h2 = computeQuadraticBasis(dx,-obj.K*dx,oaDim);
                    H = h1 - gamma* h2;
                    
                    yBuf(i, 1) = r;
                    hBuf(i, :) = H;
                    
                    % TD�덷���v�Z
                    TDError(i) = r + gamma * ...
                        evaluate(obj.Critic, {dx, -obj.K*dx}) - ...
                            evaluate(obj.Critic, {x, u});
                end
                
                % Update the critic parameters based on the batch of
                % experiences
                if (rcond(hBuf'*hBuf) > 1e-16)  % �t�s�񂪋��߂��Ȃ���
                    theta = (hBuf'*hBuf)\hBuf'*yBuf;
                    setLearnableParameterValues(obj.Critic,{theta});

                    % Derive a new gain matrix based on the new critic parameters
                    obj.K = getNewK(obj);
                    obj.KUpdate = obj.KUpdate + 1;
                    obj.KBuffer{obj.KUpdate} = obj.K;
                end
                % Caluclate TD error
                obj.TDBuffer(obj.TDBufferSize) = mean(abs(TDError));
                obj.TDBufferSize = obj.TDBufferSize + 1;
                
                % Reset the experience buffers
                obj.experienceBufferCount = 0;
                obj.experienceBuffer = cell(N, 1);
                
                % �Q�C��K�̍X�V�������ȉ��ɂȂ�����w�K�I��
                kNorm = norm((obj.KBuffer{obj.KUpdate} - ...
                    obj.KBuffer{obj.KUpdate-1}));
                if (kNorm < obj.stopLearningValue)
                    setStepMode(obj,"sim");
                end
            end
            
            % Find and return an action with exploration
            action = getActionWithExploration(obj,exp{4});
        end
        % Create critic 
        function critic = createCritic(obj)
            observeDim = obj.ObservationInfo.Dimension(1);
            actionDim = obj.ActionInfo.Dimension(1);
            n = observeDim+actionDim;
            w0 = 0.1*ones(0.5*(n+1)*n,1);
            critic = rlRepresentation(@(x,u) computeQuadraticBasis(x,u,n),w0,...
                {obj.ObservationInfo,obj.ActionInfo});
            critic.Options.GradientThreshold = 1;
            critic = critic.setLoss('mse');
        end
        % Update K from critic
        function k = getNewK(obj)
            w = getLearnableParameterValues(obj.Critic);
            w = w{1};
            observeDim = obj.ObservationInfo.Dimension(1);
            actionDim = obj.ActionInfo.Dimension(1);
            n = observeDim+actionDim;
            idx = 1;
            for r = 1:n
                for c = r:n
                    Phat(r,c) = w(idx);
                    idx = idx + 1;
                end
            end
            H  = 1/2*(Phat+Phat');
            Huu = H(observeDim+1:end,observeDim+1:end);
            Hux = H(observeDim+1:end,1:observeDim);
            if rank(Huu) == actionDim
                k = Huu\Hux;
            else
                k = obj.K;
            end
        end       
        
        % Action methods
        function action = getActionImpl(obj,Observation)
            % Given the current state of the system, return an action.
            action = -obj.K*Observation{:};
        end
 
    end
        
end

%% local function
function B = computeQuadraticBasis(x,u,n)
z = cat(1,x,u);
idx = 1;
for r = 1:n
    for c = r:n
        if idx == 1
            B = z(r)*z(c);
        else
            B = cat(1,B,z(r)*z(c));
        end
        idx = idx + 1;
    end
end
end
