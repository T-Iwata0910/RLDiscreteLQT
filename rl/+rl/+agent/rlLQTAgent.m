classdef rlLQTAgent < rl.agent.CustomAgent
    % rlLQTAgent: Creates a LQT agent.
    %
    %   agent = rlLQTAgent(CRITIC) creates a LQT agent with default
    %   options and the specified critic representation.
    %
    %   agent = rlLQTAgent(CRITIC,OPTIONS) creates a LQT agent with
    %   the specified options. To create OPTIONS, use rlLQTAgentOptions.
    %
    
    % Copyright 2020 T.Iwata 
    
    %% Public Properties
    properties (Dependent)
        % Options to configure RL agent
        AgentOptions
    end
    
    properties

        % Feedback gain
        K

        % Discount Factor
        Gamma = 0.8

        % Critic
        Critic
        
        % TD error buffer
        TDBuffer
        TDBufferSize = 1
        
        % Buffer for K
        KBuffer  
        % Number of updates for K
        KUpdate = 1

        % Number for estimator update
        EstimateNum = 20
        
        % Stop learning value
        % ゲインの更新幅がこの値以下になったら学習を終了する
        stopLearningValue
    end
    
    properties (Access = private)
        stepMode
        experienceBuffer
        experienceBufferCount = 0
    end
    
    
    %% MAIN METHODS
    methods
        % Constructor
        function obj = rlLQTAgent(Q,R,InitialK,varargin)
            % input parser
            p = inputParser;
            addParameter(p, 'StopLearningValue', 1e-5, @isnumeric);
            parse(p, varargin{:});

            % Call the abstract class constructor
            obj = obj@rl.agent.CustomAgent();

            % Set the Q and R matrices
            obj.Q = Q;
            obj.R = R;

            % Define the observation and action spaces
            obj.ObservationInfo = rlNumericSpec([size(Q,1),1]);
            obj.ActionInfo = rlNumericSpec([size(R,1),1]);

            % Create the critic representation
            obj.Critic = createCritic(obj);

            % Initialize the gain matrix
            obj.K = InitialK;
            
            % Initialize learning parameters
            obj.stopLearningValue = p.Results.StopLearningValue;

            % Initialize the log buffers
            obj.KBuffer = cell(1,1000);
            obj.KBuffer{1} = obj.K;
        end
        end
    
    %% Implementation of abstract parent protected methods
    methods (Access = protected)
        function action = getActionWithExplorationImpl(obj,Observation)
            % Given the current observation, select an action
            action = getAction(obj,Observation);
            % Add random noise to action
            num = size(obj.R,1);
            action = action + 2*randn(num,1);
        end
        % learn from current experiences, return action with exploration
        % exp = {state,action,reward,nextstate,isdone}
        function action = learnImpl(obj,exp)
            
            
            % Store in experience buffer
            obj.experienceBufferCount = obj.experienceBufferCount + 1;
            obj.experienceBuffer{obj.experienceBufferCount} = exp;
            
            
            
            % Wait N steps before updating critic parameters
            N = obj.EstimateNum;
            
            if obj.experienceBufferCount>=N
                num = size(obj.Q,1) + size(obj.R,1);
                yBuf = zeros(obj.experienceBufferCount,1);
                hBuf = zeros(obj.experienceBufferCount,0.5*num*(num+1));
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
                    h1 = computeQuadraticBasis(x,u,num);
                    h2 = computeQuadraticBasis(dx,-obj.K*dx,num);
                    H = h1 - obj.Gamma* h2;
                    
                    yBuf(i, 1) = r;
                    hBuf(i, :) = H;
                    
                    % TD誤差を計算
                    TDError(i) = r + obj.Gamma * ...
                        evaluate(obj.Critic, {dx, -obj.K*dx}) - ...
                            evaluate(obj.Critic, {x, u});
                end
                
                % Update the critic parameters based on the batch of
                % experiences
                if (rcond(hBuf'*hBuf) > 1e-16)  % 逆行列が求められない時
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
                
                % ゲインKの更新幅が一定以下になったら学習終了
                kNorm = norm((obj.KBuffer{obj.KUpdate}- ...
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
            nQ = size(obj.Q,1);
            nR = size(obj.R,1);
            n = nQ+nR;
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
            nQ = size(obj.Q,1);
            nR = size(obj.R,1);
            n = nQ+nR;
            idx = 1;
            for r = 1:n
                for c = r:n
                    Phat(r,c) = w(idx);
                    idx = idx + 1;
                end
            end
            H  = 1/2*(Phat+Phat');
            Huu = H(nQ+1:end,nQ+1:end);
            Hux = H(nQ+1:end,1:nQ);
            if rank(Huu) == nR
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
