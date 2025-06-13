PATH_CONFIG = {
    'train_data': 'data/train.csv',
    'test_data': 'data/testA.csv',
    'sample_submit': 'data/sample_submit.csv',
    'output_dir': 'outputs/',
    'model_dir': 'models/',
    'features': [
        'loanAmnt', 'term', 'interestRate', 'installment', 'employmentLength',
        'homeOwnership', 'annualIncome', 'purpose', 'postCode', 'regionCode',
        'dti', 'delinquency_2years', 'ficoRangeLow', 'openAcc', 'pubRec',
        'pubRecBankruptcies', 'revolUtil', 'totalAcc', 'applicationType'
    ] + [f'n{i}' for i in range(15)],
    'categorical_features': [
        'term', 'homeOwnership', 'purpose', 'regionCode', 'applicationType'
    ],
    'model_name': 'lgb_model.pkl',
    'result_name': 'submission.csv'
}