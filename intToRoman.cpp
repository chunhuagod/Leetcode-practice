string intToRoman(int num) {
    map<int, char> int_roman = {{1,    'I'},
                                {5,    'V'},
                                {10,   'X'},
                                {50,   'L'},
                                {100,  'C'},
                                {500,  'D'},
                                {1000, 'M'}};
    vector<int> index({1, 10, 100, 1000});
    string final;
    while (num) {
        int temp = num / index.back();
        num = num % index.back();
        while (temp) {
            if (temp == 9) {
                final.append(1, int_roman[index.back()]);
                final.append(1, int_roman[10 * index.back()]);
                break;
            }
            if (temp < 4) {
                final.append(temp, int_roman[index.back()]);
                break;
            }
            if (temp == 4) {
                final.append(1, int_roman[index.back()]);
                final.append(1, int_roman[5 * index.back()]);
                break;
            }
            final.append(1, int_roman[5 * index.back()]);
            final.append(temp - 5, int_roman[index.back()]);
            break;
        }
        index.pop_back();
    }
    return final;
}