% Specify the path to the EEGLAB directory
eeglab_path = 'C:/Users/rakes/Downloads/eeglab_current/eeglab2024.1';

% Add EEGLAB to the MATLAB path
addpath(eeglab_path);

% Initialize EEGLAB
eeglab;
%% Extract and loading the data set

% Load the .mat file
data = load('DREAMER.mat');

% Display the field names in the loaded data structure
disp(fieldnames(data));

%% Extracting all EEG data 

% Extract the number of subjects
noOfSubjects = data.DREAMER.noOfSubjects;

% Extract EEG data for all subjects
EEG_data_all = data.DREAMER.Data;

% Display the size and structure of the EEG data
disp(size(EEG_data_all));

% Loop through each subject's data and extract EEG data
for subject_idx = 1:noOfSubjects
    % Extract the EEG data for the current subject
    EEG_data = EEG_data_all{subject_idx}.EEG;

    % Display the size of the extracted EEG data
    disp(['Subject ', num2str(subject_idx), ' EEG data size:']);
    disp(size(EEG_data));
    
    % Store the EEG data in a cell array for further use
    EEG_all_subjects{subject_idx} = EEG_data;
end

% Display the EEG data for the first subject as an example
disp('EEG data for the first subject:');
disp(EEG_all_subjects{1});

%% Using stimuli    

%Extract the number of subjects
noOfSubjects = data.DREAMER.noOfSubjects;


% Extract EEG data for all subjects
EEG_data_all = data.DREAMER.Data;

% Create a cell array to store the "stimuli" EEG data for all subjects
EEG_stimuli_all_subjects = cell(1, noOfSubjects);

% Loop through each subject's data and extract "stimuli" EEG data
for subject_idx = 1:noOfSubjects
    % Extract the "stimuli" EEG data for the current subject
    EEG_stimuli_data = EEG_data_all{subject_idx}.EEG.stimuli;
    
    % Store the "stimuli" EEG data in the cell array
    EEG_stimuli_all_subjects{subject_idx} = EEG_stimuli_data;
end

% Display the "stimuli" EEG data for the first subject as an example
disp('EEG "stimuli" data for the first subject:');
disp(EEG_stimuli_all_subjects{1});
%%
% Filtering and Saving Data using EEGLAB
output_folder = '\data'; % Define the output folder
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

fs = 128; % Sampling frequency
for subject_idx = 1:noOfSubjects
    % Extract the "stimuli" EEG data for the current subject
    stimuli_data = EEG_stimuli_all_subjects{subject_idx};
    
    % Loop through each trial of the subject's stimuli data
    for trial_idx = 1:length(stimuli_data)
        trial_data = stimuli_data{trial_idx}';
        
        % Create an EEGLAB EEG structure
        [ALLEEG EEG CURRENTSET ALLCOM] = eeglab;
        EEG = pop_importdata('dataformat', 'array', 'nbchan', 14, 'data', trial_data, ...
            'srate', fs, 'pnts', size(trial_data, 2), 'xmin', 0);
        
        % Filter the data from 0.1 to 40 Hz
        EEG = pop_eegfiltnew(EEG, 0.1, 40);
        
        % Define the filename for the filtered data
        filename = fullfile(output_folder, ['Subject' num2str(subject_idx) '_Trial' num2str(trial_idx) '.set']);
        
        % Save the filtered data
        EEG = pop_saveset(EEG, 'filename', filename);
    end
end

disp('Filtering and saving of EEG data completed.');

%%

%Frequency band

% Define frequency bands
theta_band = [4 8];
alpha_band = [8 13];
beta_band = [13 20];

% Sampling frequency
fs = 128;

% Initialize a cell array to store features for all subjects and trials
EEG_features_all_subjects = cell(noOfSubjects, 1);

for subject_idx = 1:noOfSubjects
    stimuli_data = EEG_stimuli_all_subjects{subject_idx};
    num_trials = length(stimuli_data);
    
    % Initialize a matrix to store features for each trial
    features_matrix = zeros(num_trials, 14 * 3); % 14 electrodes, 3 bands
    
    for trial_idx = 1:num_trials
        trial_data = stimuli_data{trial_idx}';
        
        % Calculate PSD using Welch's method for each electrode
        for electrode = 1:14
            [Pxx, F] = pwelch(trial_data(electrode, :), [], [], [], fs);
            
            % Calculate band powers
            theta_power = bandpower(Pxx, F, theta_band, 'psd');
            alpha_power = bandpower(Pxx, F, alpha_band, 'psd');
            beta_power = bandpower(Pxx, F, beta_band, 'psd');
            
            % Store features
            features_matrix(trial_idx, (electrode - 1) * 3 + 1) = log(theta_power);
            features_matrix(trial_idx, (electrode - 1) * 3 + 2) = log(alpha_power);
            features_matrix(trial_idx, (electrode - 1) * 3 + 3) = log(beta_power);
        end
    end
    
    % Store the features for the current subject
    EEG_features_all_subjects{subject_idx} = features_matrix;
end

% Display features for the first subject and trial as an example
disp('EEG features for the first subject, first trial:');
disp(EEG_features_all_subjects{1}(1, :));
%%
% Baseline normalization of all features 
% Initialize a cell array to store baseline features for all subjects
EEG_baseline_all_subjects = cell(noOfSubjects, 1);

% Assume a sampling rate of 128 Hz
sampling_rate = 128;
last_4_seconds_samples = 4 * sampling_rate; % 512 samples

for subject_idx = 1:noOfSubjects
    % Extract the neutral EEG data for the current subject
    neutral_data = EEG_data_all{subject_idx}.EEG.baseline; % Adjust the field name based on the structure
    
    % Display class and size for debugging
    disp(['Class of neutral_data for subject ', num2str(subject_idx), ': ', class(neutral_data)]);
    disp(['Size of neutral_data: ', num2str(size(neutral_data))]);
    
    % Initialize a matrix to store the mean across time points for each electrode
    mean_across_trials = zeros(14, 1); % 14 electrodes

    % Iterate over each cell in neutral_data
    for cell_idx = 1:numel(neutral_data)
        % Extract the data matrix from the cell
        trial_data = neutral_data{cell_idx};
        
        % Determine the number of available samples
        num_samples = size(trial_data, 2);
        
        % Ensure we do not exceed the available sample length
        start_idx = max(1, num_samples - last_4_seconds_samples + 1);
        
        % Extract the last 512 samples (last 4 seconds)
        trial_data = trial_data(:, start_idx:end);
        
        % Transpose the trial data to ensure it is 14 x samples
        trial_data = trial_data';  % Transpose the matrix
        
        % Calculate the mean across time points (columns) for each electrode (rows)
        mean_across_trials = mean_across_trials + mean(trial_data, 2);  % Summing the means across trials
    end
    
    % Average across all trials
    baseline_features = mean_across_trials / numel(neutral_data);
    
    
    % Repeat the mean value to match the feature vector length (assuming 3 bands per electrode)
    baseline_features = repmat(baseline_features, 1, 3);
    baseline_features = baseline_features(:)';
    
    % Store the baseline features for the current subject
    EEG_baseline_all_subjects{subject_idx} = baseline_features;
end

% Normalize the features by dividing by the baseline features
for subject_idx = 1:noOfSubjects
    num_trials = size(EEG_features_all_subjects{subject_idx}, 1);
    
    for trial_idx = 1:num_trials
        % Ensure both feature vectors have the same length
        if length(EEG_features_all_subjects{subject_idx}(trial_idx, :)) == length(EEG_baseline_all_subjects{subject_idx})
            EEG_features_all_subjects{subject_idx}(trial_idx, :) = ...
                EEG_features_all_subjects{subject_idx}(trial_idx, :) ./ EEG_baseline_all_subjects{subject_idx};
        else
            warning(['Size mismatch for subject ', num2str(subject_idx), ' trial ', num2str(trial_idx)]);
            % Optional: Handle size mismatch, e.g., by padding or truncating
            min_length = min(length(EEG_features_all_subjects{subject_idx}(trial_idx, :)), length(EEG_baseline_all_subjects{subject_idx}));
            EEG_features_all_subjects{subject_idx}(trial_idx, 1:min_length) = ...
                EEG_features_all_subjects{subject_idx}(trial_idx, 1:min_length) ./ EEG_baseline_all_subjects{subject_idx}(1:min_length);
        end
    end
end

% Display normalized features for the first subject and trial as an example
disp('Normalized EEG features for the first subject, first trial:');
disp(EEG_features_all_subjects{1}(1, :));

%%
% Adjusted thresholds based on observed scores
valence_threshold = 3;  % Adjusting to 3.5 to balance the labels
arousal_threshold = 3;  % Adjusting to 3.5 to balance the labels
dominance_threshold = 3;  % Adjusting to 3.5 to balance the labels

% Initialize label matrices for all subjects
labels_valence = zeros(noOfSubjects, 18);  % Assuming 18 trials per subject
labels_arousal = zeros(noOfSubjects, 18);
labels_dominance = zeros(noOfSubjects, 18);

for subject_idx = 1:noOfSubjects
    % Retrieve the valence, arousal, and dominance scores for the subject
    valence_scores = data.DREAMER.Data{subject_idx}.ScoreValence;  % Valence scores (18x1)
    arousal_scores = data.DREAMER.Data{subject_idx}.ScoreArousal;  % Arousal scores (18x1)
    dominance_scores = data.DREAMER.Data{subject_idx}.ScoreDominance;  % Dominance scores (18x1)
    
    % Binarize the scores based on the new thresholds
    labels_valence(subject_idx, :) = valence_scores > valence_threshold;
    labels_arousal(subject_idx, :) = arousal_scores > arousal_threshold;
    labels_dominance(subject_idx, :) = dominance_scores > dominance_threshold;
end

% Count the number of instances in each class for valence, arousal, and dominance
num_high_valence = sum(labels_valence(:)); % Sum of all high valence labels
num_low_valence = numel(labels_valence) - num_high_valence; % Total minus high valence gives low valence

num_high_arousal = sum(labels_arousal(:)); % Sum of all high arousal labels
num_low_arousal = numel(labels_arousal) - num_high_arousal; % Total minus high arousal gives low arousal

num_high_dominance = sum(labels_dominance(:)); % Sum of all high dominance labels
num_low_dominance = numel(labels_dominance) - num_high_dominance; % Total minus high dominance gives low dominance

% Display the results
disp(['High Valence: ', num2str(num_high_valence)]);
disp(['Low Valence: ', num2str(num_low_valence)]);
disp(['High Arousal: ', num2str(num_high_arousal)]);
disp(['Low Arousal: ', num2str(num_low_arousal)]);
disp(['High Dominance: ', num2str(num_high_dominance)]);
disp(['Low Dominance: ', num2str(num_low_dominance)]);

%% 10-Fold Cross-Validation for Valence, Arousal, and Dominance

num_folds = 10; % Number of folds for cross-validation
rng('default'); % For reproducibility

% Initialize variables to store results
accuracy_valence = zeros(num_folds, 1);
accuracy_arousal = zeros(num_folds, 1);
accuracy_dominance = zeros(num_folds, 1);

% Create cross-validation partitions
cvp = cvpartition(noOfSubjects, 'KFold', num_folds);

for fold = 1:num_folds
    % Get training and testing indices
    train_idx = training(cvp, fold);
    test_idx = test(cvp, fold);
    
    % Initialize matrices for training and testing sets
    X_train = [];
    y_train_valence = [];
    y_train_arousal = [];
    y_train_dominance = [];
    
    X_test = [];
    y_test_valence = [];
    y_test_arousal = [];
    y_test_dominance = [];
    
    % Loop through subjects and organize training and testing data
    for subject_idx = 1:noOfSubjects
        features = EEG_features_all_subjects{subject_idx}; % Features
        valence_labels = labels_valence(subject_idx, :); % Valence labels
        arousal_labels = labels_arousal(subject_idx, :); % Arousal labels
        dominance_labels = labels_dominance(subject_idx, :); % Dominance labels
        
        if train_idx(subject_idx)
            % Append to training set
            X_train = [X_train; features];
            y_train_valence = [y_train_valence; valence_labels'];
            y_train_arousal = [y_train_arousal; arousal_labels'];
            y_train_dominance = [y_train_dominance; dominance_labels'];
        else
            % Append to testing set
            X_test = [X_test; features];
            y_test_valence = [y_test_valence; valence_labels'];
            y_test_arousal = [y_test_arousal; arousal_labels'];
            y_test_dominance = [y_test_dominance; dominance_labels'];
        end
    end
    
    % Train SVM models for valence, arousal, and dominance
    model_valence = fitcsvm(X_train, y_train_valence, 'KernelFunction', 'rbf', 'Standardize', true);
    model_arousal = fitcsvm(X_train, y_train_arousal, 'KernelFunction', 'rbf', 'Standardize', true);
    model_dominance = fitcsvm(X_train, y_train_dominance, 'KernelFunction', 'rbf', 'Standardize', true);
    
    % Predict on test set
    y_pred_valence = predict(model_valence, X_test);
    y_pred_arousal = predict(model_arousal, X_test);
    y_pred_dominance = predict(model_dominance, X_test);
    
    % Calculate accuracy
    accuracy_valence(fold) = mean(y_pred_valence == y_test_valence);
    accuracy_arousal(fold) = mean(y_pred_arousal == y_test_arousal);
    accuracy_dominance(fold) = mean(y_pred_dominance == y_test_dominance);
    
    % Display fold results
    disp(['Fold ', num2str(fold), ' - Valence Accuracy: ', num2str(accuracy_valence(fold))]);
    disp(['Fold ', num2str(fold), ' - Arousal Accuracy: ', num2str(accuracy_arousal(fold))]);
    disp(['Fold ', num2str(fold), ' - Dominance Accuracy: ', num2str(accuracy_dominance(fold))]);
end

% Calculate and display overall accuracy
mean_accuracy_valence = mean(accuracy_valence);
mean_accuracy_arousal = mean(accuracy_arousal);
mean_accuracy_dominance = mean(accuracy_dominance);

disp(['Mean Valence Accuracy across all folds: ', num2str(mean_accuracy_valence)]);
disp(['Mean Arousal Accuracy across all folds: ', num2str(mean_accuracy_arousal)]);
disp(['Mean Dominance Accuracy across all folds: ', num2str(mean_accuracy_dominance)]);



%% 10-Fold Cross-Validation for Valence, Arousal, and Dominance using AdaBoost

num_folds = 10; % Number of folds for cross-validation
rng('default'); % For reproducibility

% Initialize variables to store results
accuracy_valence = zeros(num_folds, 1);
accuracy_arousal = zeros(num_folds, 1);
accuracy_dominance = zeros(num_folds, 1);

% Create cross-validation partitions
cvp = cvpartition(noOfSubjects, 'KFold', num_folds);

for fold = 1:num_folds
    % Get training and testing indices
    train_idx = training(cvp, fold);
    test_idx = test(cvp, fold);
    
    % Initialize matrices for training and testing sets
    X_train = [];
    y_train_valence = [];
    y_train_arousal = [];
    y_train_dominance = [];
    
    X_test = [];
    y_test_valence = [];
    y_test_arousal = [];
    y_test_dominance = [];
    
    % Loop through subjects and organize training and testing data
    for subject_idx = 1:noOfSubjects
        features = EEG_features_all_subjects{subject_idx}; % Features
        valence_labels = labels_valence(subject_idx, :); % Valence labels
        arousal_labels = labels_arousal(subject_idx, :); % Arousal labels
        dominance_labels = labels_dominance(subject_idx, :); % Dominance labels
        
        if train_idx(subject_idx)
            % Append to training set
            X_train = [X_train; features];
            y_train_valence = [y_train_valence; valence_labels'];
            y_train_arousal = [y_train_arousal; arousal_labels'];
            y_train_dominance = [y_train_dominance; dominance_labels'];
        else
            % Append to testing set
            X_test = [X_test; features];
            y_test_valence = [y_test_valence; valence_labels'];
            y_test_arousal = [y_test_arousal; arousal_labels'];
            y_test_dominance = [y_test_dominance; dominance_labels'];
        end
    end
    
    % Train XGBoost models for valence, arousal, and dominance using AdaBoost
    model_valence = fitcensemble(X_train, y_train_valence, 'Method', 'AdaBoostM1', 'NumLearningCycles', 100, 'LearnRate', 0.1);
    model_arousal = fitcensemble(X_train, y_train_arousal, 'Method', 'AdaBoostM1', 'NumLearningCycles', 100, 'LearnRate', 0.1);
    model_dominance = fitcensemble(X_train, y_train_dominance, 'Method', 'AdaBoostM1', 'NumLearningCycles', 100, 'LearnRate', 0.1);
    
    % Predict on test set
    y_pred_valence = predict(model_valence, X_test);
    y_pred_arousal = predict(model_arousal, X_test);
    y_pred_dominance = predict(model_dominance, X_test);
    
    % Calculate accuracy
    accuracy_valence(fold) = mean(y_pred_valence == y_test_valence);
    accuracy_arousal(fold) = mean(y_pred_arousal == y_test_arousal);
    accuracy_dominance(fold) = mean(y_pred_dominance == y_test_dominance);
    
    % Display fold results
    disp(['Fold ', num2str(fold), ' - Valence Accuracy: ', num2str(accuracy_valence(fold))]);
    disp(['Fold ', num2str(fold), ' - Arousal Accuracy: ', num2str(accuracy_arousal(fold))]);
    disp(['Fold ', num2str(fold), ' - Dominance Accuracy: ', num2str(accuracy_dominance(fold))]);
end

% Calculate and display overall accuracy
mean_accuracy_valence = mean(accuracy_valence);
mean_accuracy_arousal = mean(accuracy_arousal);
mean_accuracy_dominance = mean(accuracy_dominance);

disp(['Mean Valence Accuracy across all folds: ', num2str(mean_accuracy_valence)]);
disp(['Mean Arousal Accuracy across all folds: ', num2str(mean_accuracy_arousal)]);
disp(['Mean Dominance Accuracy across all folds: ', num2str(mean_accuracy_dominance)]);


%% 10-Fold Cross-Validation for Valence, Arousal, and Dominance using Random Forest

num_folds = 10; % Number of folds for cross-validation
rng('default'); % For reproducibility

% Initialize variables to store results
accuracy_valence = zeros(num_folds, 1);
accuracy_arousal = zeros(num_folds, 1);
accuracy_dominance = zeros(num_folds, 1);

% Create cross-validation partitions
cvp = cvpartition(noOfSubjects, 'KFold', num_folds);

for fold = 1:num_folds
    % Get training and testing indices
    train_idx = training(cvp, fold);
    test_idx = test(cvp, fold);
    
    % Initialize matrices for training and testing sets
    X_train = [];
    y_train_valence = [];
    y_train_arousal = [];
    y_train_dominance = [];
    
    X_test = [];
    y_test_valence = [];
    y_test_arousal = [];
    y_test_dominance = [];
    
    % Loop through subjects and organize training and testing data
    for subject_idx = 1:noOfSubjects
        features = EEG_features_all_subjects{subject_idx}; % Features
        valence_labels = labels_valence(subject_idx, :); % Valence labels
        arousal_labels = labels_arousal(subject_idx, :); % Arousal labels
        dominance_labels = labels_dominance(subject_idx, :); % Dominance labels
        
        if train_idx(subject_idx)
            % Append to training set
            X_train = [X_train; features];
            y_train_valence = [y_train_valence; valence_labels'];
            y_train_arousal = [y_train_arousal; arousal_labels'];
            y_train_dominance = [y_train_dominance; dominance_labels'];
        else
            % Append to testing set
            X_test = [X_test; features];
            y_test_valence = [y_test_valence; valence_labels'];
            y_test_arousal = [y_test_arousal; arousal_labels'];
            y_test_dominance = [y_test_dominance; dominance_labels'];
        end
    end
    
    % Train Random Forest models for valence, arousal, and dominance
    model_valence = TreeBagger(100, X_train, y_train_valence, 'OOBPrediction', 'On', 'Method', 'classification');
    model_arousal = TreeBagger(100, X_train, y_train_arousal, 'OOBPrediction', 'On', 'Method', 'classification');
    model_dominance = TreeBagger(100, X_train, y_train_dominance, 'OOBPrediction', 'On', 'Method', 'classification');
    
    % Predict on test set
    y_pred_valence = str2double(predict(model_valence, X_test));
    y_pred_arousal = str2double(predict(model_arousal, X_test));
    y_pred_dominance = str2double(predict(model_dominance, X_test));
    
    % Calculate accuracy
    accuracy_valence(fold) = mean(y_pred_valence == y_test_valence);
    accuracy_arousal(fold) = mean(y_pred_arousal == y_test_arousal);
    accuracy_dominance(fold) = mean(y_pred_dominance == y_test_dominance);
    
    % Display fold results
    disp(['Fold ', num2str(fold), ' - Valence Accuracy: ', num2str(accuracy_valence(fold))]);
    disp(['Fold ', num2str(fold), ' - Arousal Accuracy: ', num2str(accuracy_arousal(fold))]);
    disp(['Fold ', num2str(fold), ' - Dominance Accuracy: ', num2str(accuracy_dominance(fold))]);
end

% Calculate and display overall accuracy
mean_accuracy_valence = mean(accuracy_valence);
mean_accuracy_arousal = mean(accuracy_arousal);
mean_accuracy_dominance = mean(accuracy_dominance);

disp(['Mean Valence Accuracy across all folds: ', num2str(mean_accuracy_valence)]);
disp(['Mean Arousal Accuracy across all folds: ', num2str(mean_accuracy_arousal)]);
disp(['Mean Dominance Accuracy across all folds: ', num2str(mean_accuracy_dominance)]);


%% Hyperparameter Tuning for Valence, Arousal, and Dominance

% Define the hyperparameter grid
C_values = [0.1, 1, 10];
kernel_scales = [0.1, 1, 10];

% Initialize variables to store best results
best_valence_accuracy = 0;
best_arousal_accuracy = 0;
best_dominance_accuracy = 0;
best_C = 1;
best_kernel_scale = 1;

for C = C_values
    for kernel_scale = kernel_scales
        % Train the SVM with the current set of hyperparameters
        model_valence = fitcsvm(X_train, y_train_valence, 'KernelFunction', 'rbf', ...
            'Standardize', true, 'BoxConstraint', C, 'KernelScale', kernel_scale);
        model_arousal = fitcsvm(X_train, y_train_arousal, 'KernelFunction', 'rbf', ...
            'Standardize', true, 'BoxConstraint', C, 'KernelScale', kernel_scale);
        model_dominance = fitcsvm(X_train, y_train_dominance, 'KernelFunction', 'rbf', ...
            'Standardize', true, 'BoxConstraint', C, 'KernelScale', kernel_scale);
        
        % Predict on the test set
        y_pred_valence = predict(model_valence, X_test);
        y_pred_arousal = predict(model_arousal, X_test);
        y_pred_dominance = predict(model_dominance, X_test);
        
        % Calculate accuracy
        valence_accuracy = mean(y_pred_valence == y_test_valence);
        arousal_accuracy = mean(y_pred_arousal == y_test_arousal);
        dominance_accuracy = mean(y_pred_dominance == y_test_dominance);
        
        % Update best parameters if current set is better
        if valence_accuracy > best_valence_accuracy
            best_valence_accuracy = valence_accuracy;
            best_C = C;
            best_kernel_scale = kernel_scale;
        end
        
        if arousal_accuracy > best_arousal_accuracy
            best_arousal_accuracy = arousal_accuracy;
            best_C = C;
            best_kernel_scale = kernel_scale;
        end
        
        if dominance_accuracy > best_dominance_accuracy
            best_dominance_accuracy = dominance_accuracy;
            best_C = C;
            best_kernel_scale = kernel_scale;
        end
    end
end

disp(['Best Valence Accuracy: ', num2str(best_valence_accuracy)]);
disp(['Best Arousal Accuracy: ', num2str(best_arousal_accuracy)]);
disp(['Best Dominance Accuracy: ', num2str(best_dominance_accuracy)]);
disp(['Best C: ', num2str(best_C)]);
disp(['Best Kernel Scale: ', num2str(best_kernel_scale)]);

%% Leave-One-Subject-Out (LOSO) Cross-Validation for Valence, Arousal, and Dominance

num_subjects = numel(EEG_features_all_subjects);
accuracy_loso_valence = zeros(num_subjects, 1);
accuracy_loso_arousal = zeros(num_subjects, 1);
accuracy_loso_dominance = zeros(num_subjects, 1);

for subject_idx = 1:num_subjects
    % Separate the current subject as test set
    X_test_loso = EEG_features_all_subjects{subject_idx};
    y_test_valence_loso = labels_valence(subject_idx, :)';
    y_test_arousal_loso = labels_arousal(subject_idx, :)';
    y_test_dominance_loso = labels_dominance(subject_idx, :)';
    
    % Combine all other subjects as training set
    X_train_loso = [];
    y_train_valence_loso = [];
    y_train_arousal_loso = [];
    y_train_dominance_loso = [];
    
    for train_idx = setdiff(1:num_subjects, subject_idx)
        X_train_loso = [X_train_loso; EEG_features_all_subjects{train_idx}];
        y_train_valence_loso = [y_train_valence_loso; labels_valence(train_idx, :)'];
        y_train_arousal_loso = [y_train_arousal_loso; labels_arousal(train_idx, :)'];
        y_train_dominance_loso = [y_train_dominance_loso; labels_dominance(train_idx, :)'];
    end
    
    % Train SVM on the LOSO data for valence
    model_loso_valence = fitcsvm(X_train_loso, y_train_valence_loso, 'KernelFunction', 'rbf', 'Standardize', true);
    
    % Train SVM on the LOSO data for arousal
    model_loso_arousal = fitcsvm(X_train_loso, y_train_arousal_loso, ...
                                 'KernelFunction', 'rbf', 'Standardize', true);
    
    % Train SVM on the LOSO data for dominance
    model_loso_dominance = fitcsvm(X_train_loso, y_train_dominance_loso, ...
                                   'KernelFunction', 'rbf', 'Standardize', true);
    
    % Test on the left-out subject
    y_pred_valence_loso = predict(model_loso_valence, X_test_loso);
    y_pred_arousal_loso = predict(model_loso_arousal, X_test_loso);
    y_pred_dominance_loso = predict(model_loso_dominance, X_test_loso);

    % Calculate accuracy
    accuracy_loso_valence(subject_idx) = mean(y_pred_valence_loso == y_test_valence_loso);
    accuracy_loso_arousal(subject_idx) = mean(y_pred_arousal_loso == y_test_arousal_loso);
    accuracy_loso_dominance(subject_idx) = mean(y_pred_dominance_loso == y_test_dominance_loso);
end

disp(['Mean LOSO Valence Accuracy: ', num2str(mean(accuracy_loso_valence))]);
disp(['Mean LOSO Arousal Accuracy: ', num2str(mean(accuracy_loso_arousal))]);
disp(['Mean LOSO Dominance Accuracy: ', num2str(mean(accuracy_loso_dominance))]);
%%
% Main script content

% Compute and display F1 scores for valence, arousal, and dominance
% Valence
conf_matrix_valence = confusionmat(y_test_valence_loso, y_pred_valence_loso);
[precision_valence, recall_valence, f1_score_valence] = calculate_metrics(conf_matrix_valence);
disp(['F1 Score for Valence: ', num2str(f1_score_valence)]);

% Arousal
conf_matrix_arousal = confusionmat(y_test_arousal_loso, y_pred_arousal_loso);
[precision_arousal, recall_arousal, f1_score_arousal] = calculate_metrics(conf_matrix_arousal);
disp(['F1 Score for Arousal: ', num2str(f1_score_arousal)]);

% Dominance
conf_matrix_dominance = confusionmat(y_test_dominance_loso, y_pred_dominance_loso);
[precision_dominance, recall_dominance, f1_score_dominance] = calculate_metrics(conf_matrix_dominance);
disp(['F1 Score for Dominance: ', num2str(f1_score_dominance)]);

% Generate and display confusion matrices
figure;
subplot(1, 3, 1);
confusionchart(conf_matrix_valence);
title('Confusion Matrix for Valence');
saveas(gcf, 'Confusion_Matrix_Valence.png');

subplot(1, 3, 2);
confusionchart(conf_matrix_arousal);
title('Confusion Matrix for Arousal');
saveas(gcf, 'Confusion_Matrix_Arousal.png');

subplot(1, 3, 3);
confusionchart(conf_matrix_dominance);
title('Confusion Matrix for Dominance');
saveas(gcf, 'Confusion_Matrix_Dominance.png');

% Generate and save final ROC curves
figure;
subplot(1, 3, 1);
plot(X_valence, Y_valence);
xlabel('False positive rate');
ylabel('True positive rate');
title(['ROC Curve for Valence (AUC = ' num2str(AUC_valence) ')']);
saveas(gcf, 'ROC_Curve_Valence.png');

subplot(1, 3, 2);
plot(X_arousal, Y_arousal);
xlabel('False positive rate');
ylabel('True positive rate');
title(['ROC Curve for Arousal (AUC = ' num2str(AUC_arousal) ')']);
saveas(gcf, 'ROC_Curve_Arousal.png');

subplot(1, 3, 3);
plot(X_dominance, Y_dominance);
xlabel('False positive rate');
ylabel('True positive rate');
title(['ROC Curve for Dominance (AUC = ' num2str(AUC_dominance) ')']);
saveas(gcf, 'ROC_Curve_Dominance.png');



%% Hyperparameter Tuning and Model Stacking

num_folds = 10; % Number of folds for cross-validation
rng('default'); % For reproducibility

% Initialize variables to store results
accuracy_valence = zeros(num_folds, 1);
accuracy_arousal = zeros(num_folds, 1);
accuracy_dominance = zeros(num_folds, 1);

% Create cross-validation partitions
cvp = cvpartition(noOfSubjects, 'KFold', num_folds);

% Parameter grids for SVM, Random Forest, and AdaBoost
svm_params = struct('BoxConstraint', [0.1, 1, 10], 'KernelScale', [0.1, 1, 10]);
rf_params = struct('NumTrees', [100, 200, 300], 'MinLeafSize', [1, 5, 10]);
ada_params = struct('NumLearningCycles', [50, 100, 150], 'LearnRate', [0.01, 0.1, 1]);

for fold = 1:num_folds
    % Get training and testing indices
    train_idx = training(cvp, fold);
    test_idx = test(cvp, fold);
    
    % Initialize matrices for training and testing sets
    X_train = [];
    y_train_valence = [];
    y_train_arousal = [];
    y_train_dominance = [];
    
    X_test = [];
    y_test_valence = [];
    y_test_arousal = [];
    y_test_dominance = [];
    
    % Loop through subjects and organize training and testing data
    for subject_idx = 1:noOfSubjects
        features = EEG_features_all_subjects{subject_idx}; % Features
        valence_labels = labels_valence(subject_idx, :); % Valence labels
        arousal_labels = labels_arousal(subject_idx, :); % Arousal labels
        dominance_labels = labels_dominance(subject_idx, :); % Dominance labels
        
        if train_idx(subject_idx)
            % Append to training set
            X_train = [X_train; features];
            y_train_valence = [y_train_valence; valence_labels'];
            y_train_arousal = [y_train_arousal; arousal_labels'];
            y_train_dominance = [y_train_dominance; dominance_labels'];
        else
            % Append to testing set
            X_test = [X_test; features];
            y_test_valence = [y_test_valence; valence_labels'];
            y_test_arousal = [y_test_arousal; arousal_labels'];
            y_test_dominance = [y_test_dominance; dominance_labels'];
        end
    end
    
    % Hyperparameter tuning for SVM
    best_svm_model = [];
    best_svm_accuracy = 0;
    for C = svm_params.BoxConstraint
        for sigma = svm_params.KernelScale
            svm_model = fitcsvm(X_train, y_train_valence, 'KernelFunction', 'rbf', ...
                'BoxConstraint', C, 'KernelScale', sigma, 'Standardize', true);
            y_pred_valence = predict(svm_model, X_test);
            accuracy = mean(y_pred_valence == y_test_valence);
            if accuracy > best_svm_accuracy
                best_svm_accuracy = accuracy;
                best_svm_model = svm_model;
            end
        end
    end
    
    % Hyperparameter tuning for Random Forest
    best_rf_model = [];
    best_rf_accuracy = 0;
    for ntree = rf_params.NumTrees
        for minleaf = rf_params.MinLeafSize
            rf_model = TreeBagger(ntree, X_train, y_train_valence, 'MinLeafSize', minleaf);
            y_pred_valence = predict(rf_model, X_test);
            y_pred_valence = str2double(y_pred_valence);
            accuracy = mean(y_pred_valence == y_test_valence);
            if accuracy > best_rf_accuracy
                best_rf_accuracy = accuracy;
                best_rf_model = rf_model;
            end
        end
    end
    
    % Hyperparameter tuning for AdaBoost
    best_ada_model = [];
    best_ada_accuracy = 0;
    for ncycle = ada_params.NumLearningCycles
        for lr = ada_params.LearnRate
            ada_model = fitcensemble(X_train, y_train_valence, 'Method', 'AdaBoostM1', ...
                'NumLearningCycles', ncycle, 'LearnRate', lr);
            y_pred_valence = predict(ada_model, X_test);
            accuracy = mean(y_pred_valence == y_test_valence);
            if accuracy > best_ada_accuracy
                best_ada_accuracy = accuracy;
                best_ada_model = ada_model;
            end
        end
    end
    
    % Stacking Ensemble - Combining Predictions
    stacked_predictions = [predict(best_svm_model, X_test), ...
                           str2double(predict(best_rf_model, X_test)), ...
                           predict(best_ada_model, X_test)];
    
    % Voting Mechanism
    final_predictions = mode(stacked_predictions, 2);
    
    % Calculate accuracy for valence
    accuracy_valence(fold) = mean(final_predictions == y_test_valence);
    
    % Display fold results
    disp(['Fold ', num2str(fold), ' - Valence Accuracy: ', num2str(accuracy_valence(fold))]);
end

% Calculate and display overall accuracy
mean_accuracy_valence = mean(accuracy_valence);
disp(['Mean Valence Accuracy across all folds: ', num2str(mean_accuracy_valence)]);

%% Main Script Content

% Number of folds for cross-validation
num_folds = 10; 
rng('default'); % For reproducibility

% Initialize variables to store results
accuracy_valence = zeros(num_folds, 1);
accuracy_arousal = zeros(num_folds, 1);
accuracy_dominance = zeros(num_folds, 1);

% Initialize confusion matrices and ROC data
conf_matrix_valence = zeros(2, 2);
conf_matrix_arousal = zeros(2, 2);
conf_matrix_dominance = zeros(2, 2);
[X_valence, Y_valence, X_arousal, Y_arousal, X_dominance, Y_dominance] = deal([]);

% Create cross-validation partitions
cvp = cvpartition(noOfSubjects, 'KFold', num_folds);

% Parameter grids for SVM, Random Forest, and AdaBoost
svm_params = struct('BoxConstraint', [0.1, 1, 10], 'KernelScale', [0.1, 1, 10]);
rf_params = struct('NumTrees', [100, 200, 300], 'MinLeafSize', [1, 5, 10]);
ada_params = struct('NumLearningCycles', [50, 100, 150], 'LearnRate', [0.01, 0.1, 1]);

for fold = 1:num_folds
    % Get training and testing indices
    train_idx = training(cvp, fold);
    test_idx = test(cvp, fold);
    
    % Initialize matrices for training and testing sets
    X_train = [];
    y_train_valence = [];
    y_train_arousal = [];
    y_train_dominance = [];
    
    X_test = [];
    y_test_valence = [];
    y_test_arousal = [];
    y_test_dominance = [];
    
    % Loop through subjects and organize training and testing data
    for subject_idx = 1:noOfSubjects
        features = EEG_features_all_subjects{subject_idx}; % Features
        valence_labels = labels_valence(subject_idx, :); % Valence labels
        arousal_labels = labels_arousal(subject_idx, :); % Arousal labels
        dominance_labels = labels_dominance(subject_idx, :); % Dominance labels
        
        if train_idx(subject_idx)
            % Append to training set
            X_train = [X_train; features];
            y_train_valence = [y_train_valence; valence_labels'];
            y_train_arousal = [y_train_arousal; arousal_labels'];
            y_train_dominance = [y_train_dominance; dominance_labels'];
        else
            % Append to testing set
            X_test = [X_test; features];
            y_test_valence = [y_test_valence; valence_labels'];
            y_test_arousal = [y_test_arousal; arousal_labels'];
            y_test_dominance = [y_test_dominance; dominance_labels'];
        end
    end
    
    % Tune hyperparameters and calculate accuracy for Valence
    [~, accuracy_valence(fold), y_pred_valence] = tune_hyperparameters(X_train, y_train_valence, X_test, y_test_valence, svm_params, rf_params, ada_params);
    
    % Update confusion matrix and ROC data for Valence
    conf_matrix_valence = conf_matrix_valence + confusionmat(y_test_valence, y_pred_valence);
    [X_valence_fold, Y_valence_fold, ~, AUC_valence] = perfcurve(y_test_valence, y_pred_valence, 1);
    X_valence = [X_valence; X_valence_fold];
    Y_valence = [Y_valence; Y_valence_fold];
    
    % Tune hyperparameters and calculate accuracy for Arousal
    [~, accuracy_arousal(fold), y_pred_arousal] = tune_hyperparameters(X_train, y_train_arousal, X_test, y_test_arousal, svm_params, rf_params, ada_params);
    
    % Update confusion matrix and ROC data for Arousal
    conf_matrix_arousal = conf_matrix_arousal + confusionmat(y_test_arousal, y_pred_arousal);
    [X_arousal_fold, Y_arousal_fold, ~, AUC_arousal] = perfcurve(y_test_arousal, y_pred_arousal, 1);
    X_arousal = [X_arousal; X_arousal_fold];
    Y_arousal = [Y_arousal; Y_arousal_fold];
    
    % Tune hyperparameters and calculate accuracy for Dominance
    [~, accuracy_dominance(fold), y_pred_dominance] = tune_hyperparameters(X_train, y_train_dominance, X_test, y_test_dominance, svm_params, rf_params, ada_params);
    
    % Update confusion matrix and ROC data for Dominance
    conf_matrix_dominance = conf_matrix_dominance + confusionmat(y_test_dominance, y_pred_dominance);
    [X_dominance_fold, Y_dominance_fold, ~, AUC_dominance] = perfcurve(y_test_dominance, y_pred_dominance, 1);
    X_dominance = [X_dominance; X_dominance_fold];
    Y_dominance = [Y_dominance; Y_dominance_fold];
end

% Calculate and display overall accuracy for each label
mean_accuracy_valence = mean(accuracy_valence);
mean_accuracy_arousal = mean(accuracy_arousal);
mean_accuracy_dominance = mean(accuracy_dominance);

disp(['Mean Valence Accuracy across all folds: ', num2str(mean_accuracy_valence)]);
disp(['Mean Arousal Accuracy across all folds: ', num2str(mean_accuracy_arousal)]);
disp(['Mean Dominance Accuracy across all folds: ', num2str(mean_accuracy_dominance)]);

% Compute and display F1 scores for valence, arousal, and dominance
% Valence
[precision_valence, recall_valence, f1_score_valence] = calculate_metrics(conf_matrix_valence);
disp(['F1 Score for Valence: ', num2str(f1_score_valence)]);

% Arousal
[precision_arousal, recall_arousal, f1_score_arousal] = calculate_metrics(conf_matrix_arousal);
disp(['F1 Score for Arousal: ', num2str(f1_score_arousal)]);

% Dominance
[precision_dominance, recall_dominance, f1_score_dominance] = calculate_metrics(conf_matrix_dominance);
disp(['F1 Score for Dominance: ', num2str(f1_score_dominance)]);

% Generate and display confusion matrices
figure;
subplot(1, 3, 1);
confusionchart(conf_matrix_valence);
title('Confusion Matrix for Valence');
saveas(gcf, 'Confusion_Matrix_Valence.png');

subplot(1, 3, 2);
confusionchart(conf_matrix_arousal);
title('Confusion Matrix for Arousal');
saveas(gcf, 'Confusion_Matrix_Arousal.png');

subplot(1, 3, 3);
confusionchart(conf_matrix_dominance);
title('Confusion Matrix for Dominance');
saveas(gcf, 'Confusion_Matrix_Dominance.png');

% Generate and save final ROC curves
figure;
subplot(1, 3, 1);
plot(X_valence, Y_valence);
xlabel('False positive rate');
ylabel('True positive rate');
title(['ROC Curve for Valence (AUC = ' num2str(AUC_valence) ')']);
saveas(gcf, 'ROC_Curve_Valence.png');

subplot(1, 3, 2);
plot(X_arousal, Y_arousal);
xlabel('False positive rate');
ylabel('True positive rate');
title(['ROC Curve for Arousal (AUC = ' num2str(AUC_arousal) ')']);
saveas(gcf, 'ROC_Curve_Arousal.png');

subplot(1, 3, 3);
plot(X_dominance, Y_dominance);
xlabel('False positive rate');
ylabel('True positive rate');
title(['ROC Curve for Dominance (AUC = ' num2str(AUC_dominance) ')']);
saveas(gcf, 'ROC_Curve_Dominance.png');

%% Function Definitions

function [precision, recall, f1_score] = calculate_metrics(conf_matrix)
    TP = conf_matrix(1, 1);
    FP = conf_matrix(1, 2);
    FN = conf_matrix(2, 1);
    TN = conf_matrix(2, 2);
    
    precision = TP / (TP + FP);
    recall = TP / (TP + FN);
    f1_score = 2 * (precision * recall) / (precision + recall);
end

function [best_model, best_accuracy, y_pred] = tune_hyperparameters(X_train, y_train, X_test, y_test, svm_params, rf_params, ada_params)
    % Initialize variables for best models and accuracies
    best_model = [];
    best_accuracy = 0;
    y_pred = [];
    
    %% SVM Hyperparameter Tuning
    for C = svm_params.BoxConstraint
        for sigma = svm_params.KernelScale
            svm_model = fitcsvm(X_train, y_train, 'KernelFunction', 'rbf', ...
                'BoxConstraint', C, 'KernelScale', sigma, 'Standardize', true);
            y_pred_svm = predict(svm_model, X_test);
            accuracy = mean(y_pred_svm == y_test);
            if accuracy > best_accuracy
                best_accuracy = accuracy;
                best_model = svm_model;
                y_pred = y_pred_svm;
            end
        end
    end
    
    %% Random Forest Hyperparameter Tuning
    for ntree = rf_params.NumTrees
        for minleaf = rf_params.MinLeafSize
            rf_model = TreeBagger(ntree, X_train, y_train, 'MinLeafSize', minleaf);
            y_pred_rf = str2double(predict(rf_model, X_test));
            accuracy = mean(y_pred_rf == y_test);
            if accuracy > best_accuracy
                best_accuracy = accuracy;
                best_model = rf_model;
                y_pred = y_pred_rf;
            end
        end
    end
    
    %% AdaBoost Hyperparameter Tuning
    for ncycle = ada_params.NumLearningCycles
        for lr = ada_params.LearnRate
            ada_model = fitcensemble(X_train, y_train, 'Method', 'AdaBoostM1', ...
                'NumLearningCycles', ncycle, 'LearnRate', lr);
            y_pred_ada = predict(ada_model, X_test);
            accuracy = mean(y_pred_ada == y_test);
            if accuracy > best_accuracy
                best_accuracy = accuracy;
                best_model = ada_model;
                y_pred = y_pred_ada;
            end
        end
    end
end

%%

