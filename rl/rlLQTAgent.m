function Agent = rlLQTAgent(varargin)
    % rlLQTAgent: Creates a LQT agent.
    %
    %   agent = rlLQTAgent(CRITIC) creates a LQT agent with default
    %   options and the specified critic representation.
    %
    %   agent = rlLQTAgent(CRITIC,OPTIONS) creates a LQT agent with
    %   the specified options. To create OPTIONS, use rlLQTAgentOptions.
    %
    
    % Copyright 2020 T.Iwata

Agent = rl.agent.rlLQTAgent(varargin{:});

end