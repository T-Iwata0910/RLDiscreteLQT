classdef rlLQTAgentOptions < rl.option.AgentGeneric
    % rlLQTAgentOptions: Create options for LQT control Agent.
    %
    %   OPT = rlLQTAgentOptions returns the default options for rlLQTAgent. 
    %
    %   OPT = rlLQTAgentOptions('Option1',Value1,'Option2',Value2,...) uses name/value
    %   pairs to override the default values for 'Option1','Option2',...
    %
    %   Supported options are:
    %
    % %   DiscountFactor                      Discount factor to apply to future rewards during training
    % %   StepNumPreIteration                 1イテレーションあたりのステップ数
    % %
    % %   See also: 
    
    % ver1.0.0 2020-02-11 T.Iwata Test create
    % ver1.1.0 2020-04-30 割引率を追加
    
    
    properties
        % Number for estimator update
        StepNumPerIteration
    end
    
    methods
        function obj = rlLQTAgentOptions(varargin)
            obj = obj@rl.option.AgentGeneric(varargin{:});
            parser = obj.Parser;
            
            addParameter(parser, 'StepNumPerIteration', 10);
            
            parse(parser, varargin{:});
            obj.Parser = parser;
            obj.StepNumPerIteration = parser.Results.StepNumPerIteration;
            obj.DiscountFactor =  parser.Results.DiscountFactor;
            
            parser.KeepUnmatched = false;
            parse(parser, varargin{:});
        end
        % TODO: Impriment varidate function
        % Varidate function
        function obj = set.StepNumPerIteration(obj, value)
            validateattributes(value, {'numeric'}, {'scalar', 'real', 'integer', 'positive', 'finite'}, '', 'StepNumPerIteration');
            obj.StepNumPerIteration = value;
        end
%         function obj = set.EstimateNum(obj, Value)
%             validateattributes(Value, {'scalar'})
%         end
    end
end
