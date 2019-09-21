int romanToInt(string s) {
    map<char,int> roman_map={{'I',1},{'V',5},{'X',10},{'L',50},{'C',100},{'D',500},{'M',1000}};
    int flag_max=0;
    int total=0;
    for(int i=s.size()-1;i>=0;--i)
    {
        if(roman_map[s[i]]<flag_max) {total-=roman_map[s[i]];
            continue;}
        else {
            total += roman_map[s[i]];
            flag_max=roman_map[s[i]];
            continue;
        }
    }
    return total;
}