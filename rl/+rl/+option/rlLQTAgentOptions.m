classdef rlLQTAgentOptions < rl.option.AgentGeneric
    % rlLQTAgentOptions: Create options for LQT control Agent.
    %
    %   OPT = rlLQTAgentOptions returns the default options for rlLQTAgent. 
    %
    %   OPT = rlLQTAgentOptions('Option1',Value1,'Option2',Value2,...) uses name/value
    %   pairs to override the default values for 'Option1','Option2',...
    %
    %   TODO: サポートしている引数を追加
    %   Supported options are:
    %
    % %   EpsilonGreedyExploration            Parameters for Epsilon Greedy exploration
    % %       Epsilon                         Probability threshold for agent to either randomly
    % %                                       select a valid action or select the action that 
    % %                                       maximizes the state-action value function
    % %       EpsilonMin                      Minimum value of Epsilon
    % %       EpsilonDecay                    Decay rate of Epsilon when Epsilon is updated
    % %   SampleTime                          Sample time of the agent
    % %   DiscountFactor                      Discount factor to apply to future rewards during training
    % %
    % %   See also: rlSARSAAgent, rlDDPGAgentOptions, rlPGAgentOptions, rlACAgentOptions

    % Copyright 2020 T.Iwata
    
    properties
        % Number for estimator update
        EstimateNum
    end
    
    methods
        function obj = rlLQTAgentOptions(varargin)
            obj = obj@rl.option.AgentGeneric(varargin{:});
            parser = obj.Parser;
            addParameter(parser, 'EstimateNum', 10), ...
             
            parse(parser, varargin{:})
            obj.Parser = parser;
            obj.EstimateNum = parser.Results.EstimateNum;
            
            parser.KeepUnmatched = false;
            parse(parser, varargin{:});
        end
        % TODO: Impriment varidate function
%         function obj = set.EstimateNum(obj, Value)
%             validateattributes(Value, {'scalar'})
%         end
    end
end
