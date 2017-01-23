

category = 'allegro';
experimentName = 'test';
numTrials = 1;
numIterations = 5; % 15

numberOfFeatures = 500;

fingerList = [1,2];
useSensors = 0;

jointIdxList = [];
taskIdxList = [];
sensorIdxList = [];
for f=fingerList
    if f==1
        jointIdxList = [jointIdxList; 1;2;3];
        if useSensors
            sensorIdxList = [sensorIdxList; [0:17]'];
        end
    elseif f==2
        jointIdxList = [jointIdxList; 5;6;7];
        if useSensors
            sensorIdxList = [sensorIdxList; [18:35]'];
        end
    elseif f==4
        jointIdxList = [jointIdxList; 14;15];
        if useSensors
            sensorIdxList = [sensorIdxList; [54:71]'];
        end
    end
end

jointIdxList = [1;5];


configuredTask = Experiments.Tasks.AllegroTask(1+numel(jointIdxList) + numel(taskIdxList)+numel(sensorIdxList), numel(jointIdxList));

%%
configuredLearner = Experiments.Learner.StepBasedRKHSREPS('RKHSREPSPeriodic');

% feature configurator
configuredFeatures = Experiments.Features.FeatureRBFKernelStatesPeriodicNew;
configuredPolicyFeatures = Experiments.Features.PolicyFeatureGenerator;
%configuredActionFeatures = Experiments.Features.FeatureRBFKernelActionsProdNew;
configureModelKernel = Experiments.Features.ModelFeatures;

% action policy configurator
configuredPolicy = Experiments.ActionPolicies.SingleGaussianProcessPolicyConfiguratorNew;
%configuredPolicy = Experiments.ActionPolicies.MCMCActionPolicyConfigurator('MCMCpolicy');

evaluationCriterion = Experiments.EvaluationCriterion();
evaluationCriterion.setSaveIterationModulo(1);
evaluationCriterion.setSaveNumDataPoints(100);
evaluationCriterion.registerEvaluator(Evaluator.ReturnEvaluatorNewSamplesAverage());
%evaluationCriterion.registerEvaluator(Evaluator.ReturnEvaluatorEvaluationSamplesAverage());

% new samples per iteration | start samples | max number of samples per
% iteration
evaluateNRollouts = {...
    { 'numSamplesEpisodes','numInitialSamplesEpisodes','maxSamples'},...
    {
    % [3, 3, 3];
%       [3, 9,9]; ...
%       [5, 15, 15]; ...
% [6, 18 ,18]; ...
      %[10, 10, 10]; ...
%       [10, 30, 30]; ...
      %[30, 30, 30]; ...
%       [10, 30, 30]; ...
%       [15, 45, 45]; ...
%         [20, 60, 60]; ...
        [30, 90, 90]; ...
%     [20, 40, 40]; ...
    }};

evaluateMixing = {...
     {'epsilonMixing'},...
     { 0.9; %0.8; %0.5; 0.3; 0.1;  0; %1; 0.5; 0.3; 0; %0.03; ...
       }};

% simulation timesteps
evaluateDt = {...
     {'dt'},...
     { 0.05; %0.03; 0.01 ...
       }};

% 1 = one timestep delay of action
 evaluateDelay = {...
     {'delay'},...
     { 0; %0.5; %0; 0.5; 0.75; 1; 2; %0.5; 0.7; 1.0; %2.0; ...
       }};



allEvals = {evaluateDt, evaluateMixing,evaluateDelay, evaluateNRollouts};


allNames = cellfun(@(x) x{1}   , allEvals,'UniformOutput', false);
allNames = [allNames{:}];
index = cellfun(@(p) (1:numel(p{2})), allEvals, 'UniformOutput', false);
[index{:}] = ndgrid(index{:});
allVals = cellfun(@(x) x{2}   , allEvals,'UniformOutput', false);

combs = cellfun(@(p,idx) p(idx), ...
                   allVals, index, 'UniformOutput', false);

combslist = cellfun(@(c) c(:),  combs, 'UniformOutput', false);

allVals = num2cell(cell2mat(horzcat(combslist{:})));

s = Common.Settings();
%s.setProperty('GPInitializer', @(dm, out, in,i,trial) Distributions.Gaussian.GaussianLinearInFeaturesQuadraticCovariance(dm, out, in, 'BayesLinear'));
%s.setProperty('GPLearnerInitializer', @Learner.SupervisedLearner.BayesianLinearHyperLearnerCV);

%s.setProperty('GPInitializer', @(dm, out, in,i,trial) Distributions.Gaussian.Gaussian(dm,in,'GaussianMCMC'));
s.setProperty('GPInitializer', @(dm, out, in,i,trial) Distributions.Gaussian.GaussianActionContextPolicyMCMC(dm,in,'GaussianMCMC'));

%s.setProperty('GPLearnerInitializer', @(dm, ap, wt, in, out) Learner.SupervisedLearner.BayesianLinearHyperLearnerCV_MCMC(dm, ap, wt, in, out, 'trajectory'));
s.setProperty('GPLearnerInitializer',  @(dm, fa, wn, in,on) Learner.SupervisedLearner.BayesianLinearPolicyLearner_LDS(dm, fa, wn, 'policyFeatures', 'actions'  ) );
%s.setProperty('modelLearner', @(trial) Learner.ModelLearner.SampleModelLearner(trial.dataManager, ...
%               ':', trial.stateFeatures));


%s.setProperty('paramRange', [ones(1,16), 0]); could be used to set range
%of context vars

% vectors of the VREP sensors
sensor_positions = [0.22252093255519867, 0.0, 0.9749279022216797, -0.11126046627759933, 0.19270877540111542, 0.9749279022216797, -0.11126046627759933, -0.19270877540111542, 0.9749279022216797, 0.6234897971153259, 0.0, 0.7818315029144287, 0.38873952627182007, 0.48746395111083984, 0.7818315029144287, 0.38873952627182007, -0.48746395111083984, 0.7818315029144287, 0.9009688496589661, 0.0, 0.4338837265968323, 0.7579432725906372, 0.4871005415916443, 0.4338837265968323, 0.37427598237991333, 0.819550096988678, 0.4338837265968323, 0.37427598237991333, -0.819550096988678, 0.4338837265968323, 0.7579432725906372, -0.4871005415916443, 0.4338837265968323, 1.0, 0.0, 6.123234262925839e-17, 0.8660253882408142, 0.5, 6.123234262925839e-17, 0.5, 0.8660253882408142, 6.123234262925839e-17, 0.5, -0.8660253882408142, 6.123234262925839e-17, 0.8660253882408142, -0.5, 6.123234262925839e-17, 0.7579432725906372, 0.4871005415916443, -0.4338837265968323, 0.7579432725906372, -0.4871005415916443, -0.4338837265968323];

% ##################################################################################################
% PROVIDED DEMONSTRATIONS
% ##################################################################################################

% REAL HAND - gaiting
% ##################################################################################################
% index grip
goalpos1.joints = [ 0.019834    1.58454   0.237059   -6.5e-05   0.125772    1.29196   0.250428   0.019861          0   -4.9e-05 -0.0003874   -6.5e-05      1.396   0.019898   0.456399  -0.108956];
goalpos1.sensors = [ 251.16   222.9  105.72  122.68  184.46   131.2  165.48   53.66   53.58   41.32   45.12   71.82   60.34    84.1  111.96  107.68  131.74   43.94  108.42     139   79.62   53.28   29.84   27.44   28.12   29.82   35.42    43.8    42.3   44.08   40.16    58.7   38.04   39.52    47.2   42.06   54.64   35.94   36.34   46.74   21.46    2.48    4.14    4.04    4.68     5.6     9.1    4.48    4.64    4.26   12.92    5.74       4    8.02    9.64    9.42   13.74    4.58    9.88    18.7  334.74 -185.86  -62.68  -21.28  -33.44   10.18   11.24  -15.44   -74.3   32.12    -3.6  318.52   96.42   98.36   194.9   94.24  157.36    6.62    40.6   76.34];

% 3 finger grip
goalpos2.joints = [ 0.019834    1.53361   0.237059   -6.5e-05   0.125772    1.57189   0.250428   0.019861          0   -4.9e-05 -0.0003874   -6.5e-05      1.396   0.019898   0.456399  -0.108956];
goalpos2.sensors = [ 214.8 265.94 130.06 131.58 196.02    131  168.4  40.82   72.9  -13.2  28.52 -124.3 -76.48   -2.4  22.54   53.1  69.72  15.46  85.86 114.94 235.46 215.78  74.04  75.56 116.66  73.32   96.3   -5.3   8.66   -6.1   6.14  97.44   49.4  62.76  93.98   71.5  97.48  26.16   64.6   79.1  19.14    6.5   4.82   3.68    4.8   3.42    7.5   7.88   6.64   5.94  13.48    8.6   4.02   6.84    8.2   6.82   10.9   4.72   7.26  14.74 447.22   50.1    8.4  27.92  54.66  45.86  63.14 -61.74  -69.6 -40.36  -42.9 270.92  71.24  88.48 189.42     93 154.62  -8.42  51.48  88.74];

% middle grip
goalpos3.joints = [ 0.019834    1.24121   0.237059   -6.5e-05   0.125772       1.61   0.250428   0.019861          0   -4.9e-05 -0.0003874   -6.5e-05      1.396   0.019898   0.456399  -0.108956];
goalpos3.sensors = [ 74.46  78.76     60  62.18  69.46   68.2  79.94  71.64  70.16  69.18  69.04  63.88  55.76  61.04   70.9  70.92  82.98  62.84   69.2   86.8 198.72 312.88  103.3  93.84 150.44  88.06 124.22  -7.68   52.6  -54.6  21.06 -190.3 -38.22  -3.32 -17.86  28.08  33.88  32.14  58.56  75.48   22.7   2.66   3.92    4.1   5.44   5.54   10.3   3.84   3.98    3.4  12.18    4.8    3.3   7.24   9.26   8.84  13.52   3.74   9.02  17.66  291.1   17.1   9.92  18.86  28.58  34.22  48.76   4.02   1.24   6.36    7.9 148.94  39.02  49.74  103.4  57.66  96.86  11.56  37.04  71.76];

goalPositions = {goalpos1, goalpos2, goalpos3};
% goalPositions = {goalpos1, goalpos2};

%##################################################################################################


% SIMULATION
% BAR FINGERGAITING
%##################################################################################################
% goalpos1.joints  = [    -0.10612      1.17746     0.409481     0.409668     0.066166     0.948643     0.409542     0.448639            0 -4.89473e-05 -0.000387406 -6.50167e-05      1.83402    -0.179961      0.25387     0.805314];
% goalpos1.sensors  = [0.0415122        0        0   1.2449  0.26319 0.315699  2.43564  1.07705        0        0  1.15511   1.3465 0.685661        0        0 0.749767        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0   1.5426 0.491073 0.648351   2.2085 0.864281  1.39355  1.06836 0.490653        0 0.175931 0.911587        0        0        0        0        0        0        0];
%
% goalpos2.joints  = [   -0.106151      1.17744     0.409369      0.40967    0.0661587      1.17737     0.409407     0.461326            0 -4.89712e-05 -0.000387335 -6.50167e-05      1.83381    -0.179998     0.253194     0.805375];
% goalpos2.sensors  = [        0         0         0    1.5478  0.301195         0   3.43755   1.97918         0         0   1.41224   2.60889   1.66715         0         0   1.14234 0.0858056         0         0         0         0   1.92897  0.334863  0.344174   3.99432   1.86694         0         0   1.88099   2.51951    1.3433         0         0   1.35542         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0   5.87832   2.60779   2.62678   6.79686   3.74614   3.80484   2.77791   1.77524         0         0   1.81815         0         0         0         0         0         0         0];
%
% goalpos3.joints  = [   -0.106133      0.97411     0.409543      0.40971    0.0661417      1.17746     0.409429     0.461358            0  -4.8995e-05  -0.00038743 -6.50167e-05      1.83369    -0.180013     0.253851     0.805298];
% goalpos3.sensors  = [        0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0   1.27187  0.510294         0   2.49611   1.79794  0.118828         0  0.840059   1.73386   1.37933         0         0  0.546724 0.0753519         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0         0     2.044    1.1271  0.817472   2.15089   1.80875   0.86607  0.837963  0.881152  0.309634         0  0.214973         0         0         0         0         0         0         0];
%
% % goalPositions = {goalpos1, goalpos2, goalpos3};
% goalPositions = {goalpos1, goalpos2};

%##################################################################################################

% BOTTLE ROLLING
%##################################################################################################
% index and thumb
% goalpos1.sensors = [0        0        0 0.790674        0        0  3.86224  2.35824        0        0  1.42106  5.64138  3.37495        0        0  2.20757  1.53858 0.713476        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0  7.54419  5.05684  5.00751  3.70131  3.06525  2.98818        0        0        0        0        0        0        0        0        0        0        0        0 ];
% goalpos1.joints  = [6.12497e-05      1.05644     0.409581     0.422679            0 -4.88997e-05  -0.00038743 -6.50167e-05            0  -4.8995e-05 -0.000387311 -6.50167e-05      1.83358    -0.180084   -0.0338805      1.36111 ];
% goalpos2.sensors = [1.63932        0        0  4.83411  2.34435  1.36066  5.22895  3.59639 0.301473        0  2.36159   2.0597  1.34295        0        0 0.467883        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0  5.38705  2.36709  2.34181  6.41822  3.54917  3.47009  2.67519  1.76248        0        0  1.70412        0        0        0        0        0        0        0];
% goalpos2.joints  = [3.09944e-06     0.851239     0.661528     0.643544            0 -4.89235e-05 -0.000387383 -6.50644e-05            0 -4.89712e-05 -0.000387406 -6.50883e-05      1.83377    -0.180015    0.0133489      1.14947];
% goalpos3.sensors = [6.49887  3.82918  3.31978  4.84978  3.97611  2.74541  1.21469  1.09573        0        0 0.278416        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0  1.38416        0        0  4.96219  2.47249        0        0  2.41959  6.32196  3.15634        0        0  3.09452  1.00551 0.962704];
% goalpos3.joints  = [2.23637e-05     0.582663      1.20519     0.539697            0 -4.90189e-05 -0.000387406 -6.50167e-05            0 -4.90427e-05 -0.000387383 -6.50883e-05      1.83367    -0.180065     0.375691      0.26199];

% % only THUMB
% goalpos1.sensors = [0        0        0 0.790674        0        0  3.86224  2.35824        0        0  1.42106  5.64138  3.37495        0        0  2.20757  1.53858 0.713476        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0  7.54419  5.05684  5.00751  3.70131  3.06525  2.98818        0        0        0        0        0        0        0        0        0        0        0        0 ];
% goalpos1.joints  = [6.12497e-05      1.05644     0.409581     0.422679            0 -4.88997e-05  -0.00038743 -6.50167e-05            0  -4.8995e-05 -0.000387311 -6.50167e-05      1.83358    -0.180084   -0.0338805      1.36111 ];
% goalpos2.sensors = [1.63932        0        0  4.83411  2.34435  1.36066  5.22895  3.59639 0.301473        0  2.36159   2.0597  1.34295        0        0 0.467883        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0  5.38705  2.36709  2.34181  6.41822  3.54917  3.47009  2.67519  1.76248        0        0  1.70412        0        0        0        0        0        0        0];
% goalpos2.joints  = [3.09944e-06     0.851239     0.661528     0.643544            0 -4.89235e-05 -0.000387383 -6.50644e-05            0 -4.89712e-05 -0.000387406 -6.50883e-05      1.83377    -0.180015    0.0133489      1.14947];
% goalpos3.sensors = [6.49887  3.82918  3.31978  4.84978  3.97611  2.74541  1.21469  1.09573        0        0 0.278416        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0  1.38416        0        0  4.96219  2.47249        0        0  2.41959  6.32196  3.15634        0        0  3.09452  1.00551 0.962704];
% goalpos3.joints  = [2.23637e-05     0.582663      1.20519     0.539697            0 -4.90189e-05 -0.000387406 -6.50167e-05            0 -4.90427e-05 -0.000387383 -6.50883e-05      1.83367    -0.180065     0.375691      0.26199];
%
% % only INDEX
% goalpos1.sensors = [0        0        0 0.790674        0        0  3.86224  2.35824        0        0  1.42106  5.64138  3.37495        0        0  2.20757  1.53858 0.713476        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0  7.54419  5.05684  5.00751  3.70131  3.06525  2.98818        0        0        0        0        0        0        0        0        0        0        0        0 ];
% goalpos1.joints  = [6.12497e-05      1.05644     0.409581     0.422679            0 -4.88997e-05  -0.00038743 -6.50167e-05            0  -4.8995e-05 -0.000387311 -6.50167e-05      1.83358    -0.180084   -0.0338805      1.36111 ];
% goalpos2.sensors = [1.63932        0        0  4.83411  2.34435  1.36066  5.22895  3.59639 0.301473        0  2.36159   2.0597  1.34295        0        0 0.467883        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0  5.38705  2.36709  2.34181  6.41822  3.54917  3.47009  2.67519  1.76248        0        0  1.70412        0        0        0        0        0        0        0];
% goalpos2.joints  = [3.09944e-06     0.851239     0.661528     0.643544            0 -4.89235e-05 -0.000387383 -6.50644e-05            0 -4.89712e-05 -0.000387406 -6.50883e-05      1.83377    -0.180015    0.0133489      1.14947];
% goalpos3.sensors = [6.49887  3.82918  3.31978  4.84978  3.97611  2.74541  1.21469  1.09573        0        0 0.278416        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0  1.38416        0        0  4.96219  2.47249        0        0  2.41959  6.32196  3.15634        0        0  3.09452  1.00551 0.962704];
% goalpos3.joints  = [2.23637e-05     0.582663      1.20519     0.539697            0 -4.90189e-05 -0.000387406 -6.50167e-05            0 -4.90427e-05 -0.000387383 -6.50883e-05      1.83367    -0.180065     0.375691      0.26199];

% goalPositions = { goalpos1, goalpos2, goalpos3};
% goalPositions = {goalpos1, goalpos3};


%##################################################################################################


s.setProperty('goalPositions',goalPositions);
s.setProperty('jointIdxList', jointIdxList);
s.setProperty('taskIdxList', taskIdxList);
s.setProperty('sensorIdxList', sensorIdxList);

safeatures = @(trial) FeatureGenerators.IndexedFourierKernelFeatures(trial.dataManager, numel(goalPositions), trial.modelKernel, trial.maxNumberKernelSamples, {{'states','actions'}}, '~stateactionFeatures');

% #########################################################################################
% s.setProperty('modelLearner',            @(trial) Learner.ModelLearner.FeatureModelLearnernew(trial.dataManager, ...
%             ':', trial.stateFeatures,...
%             trial.nextStateFeatures,safeatures(trial)));

s.setProperty('modelLearner', @(trial) Learner.ModelLearner.SampleModelLearner(trial.dataManager, ...
               ':', trial.stateFeatures));
% #########################################################################################

experiment = Experiments.ExperimentFromConfigurators(category, ...
    {configuredTask, configuredFeatures, configureModelKernel, ...
    configuredPolicyFeatures,configuredPolicy, configuredLearner}, evaluationCriterion, numIterations);

experiment.defaultSettings.policyFeatureGenerator=@(trial_) FeatureGenerators.IndexedFourierKernelFeatures(trial_.dataManager, numel(goalPositions), copy(trial_.stateKernel), trial_.maxNumberKernelSamples, 'states', '~policyFeatures');
experiment.defaultSettings.stateFeatures=@(trial_) FeatureGenerators.IndexedFourierKernelFeatures(trial_.dataManager,  numel(goalPositions), trial_.stateKernel, trial_.maxNumberKernelSamples, 'states', '~stateFeatures');
experiment.defaultSettings.nextStateFeatures=@(trial_) FeatureGenerators.IndexedFourierKernelFeatures(trial_.dataManager, numel(goalPositions), trial_.stateKernel, trial_.maxNumberKernelSamples, 'nextStates', '~nextStateFeatures');
experiment.defaultSettings.policyInputVariables={'policyFeatures'};

experiment.defaultSettings.stateKernel = @(trial) Kernels.ExponentialQuadraticKernel(trial.dataManager, (configuredTask.dimstate -1), 'stateKernel');
experiment.defaultSettings.modelKernel_s = @(trial) Kernels.ExponentialQuadraticKernel(trial.dataManager, (configuredTask.dimstate -1), 'ModelStates');
experiment.defaultSettings.modelKernel = @(trial, numStates, numActions) Kernels.ProductKernel(trial.dataManager, numStates + numActions -1, {trial.modelKernel_s, trial.modelKernel_a}, ...
                {1:numStates-1, (numStates -1 + 1):(numStates -1 + numActions)}, 'ModelKernel');

experiment.defaultSettings.policyNoise = 0.0002^2; % 0.0005

%experiment.defaultSettings.initWidth=[0.2 0.5];

%experiment.defaultSettings.restrictToRange=false;
%experiment.defaultSettings.ParameterMapBayesLinearOptimizerCtxtWAc=[1 1 1 1];
% experiment.defaultSettings.RKHSparams_V = [0.2,0.4]; % bandwidths, state action, regul  + hardcorde - optimized
% experiment.defaultSettings.RKHSparams_ns = -[0.02, 0.4, 0.2 ];
experiment.defaultSettings.RKHSparams_V = [0.2]; % 0.4

experiment.defaultSettings.policyBandwidth=1;
experiment.defaultSettings.ExponentialQuadraticKernelUseARDstateKernel=false;
experiment.defaultSettings.ExponentialQuadraticKernelUseARDModelStates=false;
experiment.defaultSettings.ExponentialQuadraticKernelUseARDModelActions=false;
experiment.defaultSettings.resetProbTimeSteps=0.1;

%gridsize = [5,5];
%experiment.defaultSettings.policyFeatureGenerator=@(trial_) FeatureGenerators.LinearFeatures(trial_.dataManager, 'states',':');
%experiment.defaultSettings.stateFeatures= @(trial_) FeatureGenerators.RadialBasisFeatures(trial_.dataManager, {{'states'}}, trial_.stateKernel, ':',gridsize);
%experiment.defaultSettings.nextStateFeatures= @(trial_) FeatureGenerators.RadialBasisFeatures(trial_.dataManager, {{'nextStates'}}, trial_.stateKernel,':',gridsize);
%experiment.defaultSettings.bandwidth = [0.5 5];


%experiment.defaultSettings.policyFeatureGenerator=@(trial_) FeatureGenerators.RadialBasisFeatures(trial_.dataManager, {{'states'}}, copy(trial_.stateKernel), ':',[9 7]);


%experiment.defaultSettings.stateFeatureHyperParameters = [0.6 5];

experiment.defaultSettings.maxNumberKernelSamples = numberOfFeatures;
%experiment.defaultSettings.useStateFeaturesForPolicy= false;.
%experiment.defaultSettings.RKHSparams_V= [-1e-2 -0.6 -5];
%experiment.defaultSettings.RKHSparams_ns=[-1e-2 -0.6 -5 -50] ;
experiment.defaultSettings.tolSF=0.0001;
experiment.defaultSettings.epsilonAction= 0.5;
%experiment.defaultSettings.numSamplesEvaluation= 100;
%experiment.defaultSettings.GPVarianceNoiseFactorActions=1/sqrt(2) ;
%experiment.defaultSettings.GPVarianceFunctionFactor=1/sqrt(2) ;
%experiment.defaultSettings.HyperParametersOptimizerBayesLinearOptimizerActions='FMINUNC';
%experiment.defaultSettings.HyperParametersOptimizerGaussianProcessActions='FMINUNC';
%experiment.defaultSettings.HyperParametersOptimizerBayesLinearOptimizerCtxtWAc='FMINUNC';

experiment.defaultSettings.initSigmaActions = 0.5;

%TODO for BayesianLinearPolicyLearnerLDS
experiment.defaultSettings.individualCov = false;
experiment.defaultSettings.initSigmaCtxtWAc= 0.5;
experiment.defaultSettings.initSigmaParameters= 0.5; %0.025;
experiment.defaultSettings.initSigmaStrategy= 'param';

%experiment.defaultSettings.Noise_std = 0;
%warning('changed initSigma behavior, check it!')

experiment = Experiments.Experiment.addToDataBase(experiment);

experiment.addEvaluationCollection(allNames, allVals, numTrials );

experiment.startLocal();
%experiment.startBatch(100, 2000, '20:00:00');
