function env = rlDiscreteLQTEnv(Ad, Bd, Cd, Fd, Q, R, varargin)

env = rl.env.rlDiscreteLQTENv(Ad, Bd, Cd, Fd, Q, R, varargin{:});

end