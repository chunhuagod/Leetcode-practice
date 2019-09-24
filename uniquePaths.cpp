int uniquePaths(int m, int n) {
    int mm=max(m,n),nn=min(m,n);
    if(nn==1) return 1;
    vector<int> vec(mm,1);
    for(int i=2;i<nn;++i){
        for(int j=0;j<vec.size();++j){
            vector<int>::iterator s(vec.begin()+j);
            *s=accumulate(s,vec.end(),0);}
    }
    int res=accumulate(vec.cbegin(),vec.cend(),0);
    return res;
}