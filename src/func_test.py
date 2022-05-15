def func_test(mode, arg1, arg2):
    if mode == 'manual':
        test_int = arg1
    else:
        test_int = arg2

    return test_int

if __name__ == '__main__':
    print(func_test('manual', arg1=1))
