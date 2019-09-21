int maxArea(vector<int>& height) {
    int i=0,j=height.size()-1;
    int max_area=(j-i)*min(height[i],height[j]);
    int flag=0;
    for(;i<j;){
        max_area=max(max_area,min(height[i],height[j])*(j-i));
        if(height[j]>height[i]) ++i;
        else --j;
    }
    return max_area;
}