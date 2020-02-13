function Agent = rlLQTAgent(varargin)
    % rlLQTAgent: Creates a LQT agent.
    %
    %   agent = rlLQTAgent(CRITIC) creates a LQT agent with default
    %   options and the specified critic representation.
    %
    %   agent = rlLQTAgent(CRITIC,OPTIONS) creates a LQT agent with
    %   the specified options. To create OPTIONS, use rlLQTAgentOptions.
    %
    
    % ver1.0.0 2020-02-11 T.Iwata Test create

Agent = rl.agent.rlLQTAgent(varargin{:});

end