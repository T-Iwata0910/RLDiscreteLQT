classdef LQTAgent < rl.agent.CustomAgent    
    %% Public Properties
    properties
        % Q
        Q

        % R
        R

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
        EstimateNum = 21
        
        % Stop learning value
        % ゲインの更新幅がこの値以下になったら学習を終了する
        stopLearningValue
    end
    
    properties (Access = private)
        stepMode
        Counter = 1
        YBuffer
        HBuffer 
    end
    
    
    %% MAIN METHODS
    methods
        % Constructor
        function obj = LQTAgent(Q,R,InitialK,varargin)
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

            % Initialize the experience buffers
            obj.YBuffer = zeros(obj.EstimateNum,1);
            num = size(Q,1) + size(R,1);
            obj.HBuffer = zeros(obj.EstimateNum,0.5*num*(num+1));
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
            % Parse the experience input
            x = exp{1}{1};
            u = exp{2}{1};
            r = exp{3};
            dx = exp{4}{1};            
            num = size(obj.Q,1) + size(obj.R,1);
            
            % Caluclate TD error
            TDError = r + obj.Gamma * evaluate(obj.Critic, {dx, -obj.K*dx}) ...
                - evaluate(obj.Critic, {x, u});
            obj.TDBuffer(obj.TDBufferSize) = TDError;
            obj.TDBufferSize = obj.TDBufferSize + 1;
            
            
            % Wait N steps before updating critic parameters
            N = obj.EstimateNum;
            % In the linear case, critic evaluated at (x,u) is Q1 = theta'*h1,
            % critic evaluated at (dx,-K*dx) is Q2 = theta'*h2. The target
            % is to obtain theta such that Q1 - gamma*Q2 = y, that is,
            % theta'*H = y. Following is the least square solution.
            h1 = computeQuadraticBasis(x,u,num);
            h2 = computeQuadraticBasis(dx,-obj.K*dx,num);
            H = h1 - obj.Gamma* h2;
            if obj.Counter<=N
                obj.YBuffer(obj.Counter) = r;
                obj.HBuffer(obj.Counter,:) = H;
                obj.Counter = obj.Counter + 1;
            else
                % Update the critic parameters based on the batch of
                % experiences
                H_buf = obj.HBuffer;
                y_buf = obj.YBuffer;
                if (rcond(H_buf'*H_buf) > 1e-16)  % 逆行列が求められない時
                    theta = (H_buf'*H_buf)\H_buf'*y_buf;
                    setLearnableParameterValues(obj.Critic,{theta});

                    % Derive a new gain matrix based on the new critic parameters
                    obj.K = getNewK(obj);
                    obj.KUpdate = obj.KUpdate + 1;
                    obj.KBuffer{obj.KUpdate} = obj.K;
                end
                
                % Reset the experience buffers
                obj.Counter = 1;
                obj.YBuffer = zeros(N,1);
                obj.HBuffer = zeros(N,0.5*num*(num+1));
                
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
