function Options = rlLQTAgentOptions(varargin)
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
% %   StepNumPreIteration                 1�C�e���[�V����������̃X�e�b�v��
% %
% %   See also: 

% ver1.0.0 2020-02-11 T.Iwata Test create
% ver1.1.0 2020-04-30 ��������ǉ�

Options = rl.option.rlLQTAgentOptions(varargin{:});

end