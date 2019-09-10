    string convert(string s, int numRows) {
		string res;
        if (numRows==1) return s;
        for(int i=0;i<numRows;++i){
			int index=i;
			if((i==0)||i==(numRows-1)) {
				while(index<s.size()) {
					res.append(s,index,1);
					index+=2*numRows-2;
					}
				continue;
			}
			while(index<s.size()){
				res.append(s,index,1);
				if((index+2*numRows-2-2*i)<s.size()){
					res.append(s,index+(2*numRows-2-2*i),1);	
				}
				index+=2*numRows-2;
			}
		}
		return res;
    }