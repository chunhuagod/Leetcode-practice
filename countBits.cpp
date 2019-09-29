vector<int> countBits(int num) {
    vector<int> res(1,0);
    if(num==0) return res;
    res.push_back(1);
    if(num==1) return res;
    int flag=1;
    int index=2;
    while(index<=num){
        if(index>=2*flag) {flag=2*flag;}
        res.push_back(1+res[index-flag]);
        ++index;
    }
    return res;
}