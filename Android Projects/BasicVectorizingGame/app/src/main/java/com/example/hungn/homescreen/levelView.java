package com.example.hungn.homescreen;

import android.content.Context;
import android.util.AttributeSet;
import android.view.LayoutInflater;
import android.view.View;
import android.widget.LinearLayout;
import android.widget.RelativeLayout;

public class levelView extends View {
    public levelView(Context context, AttributeSet attrs){
        super(context, attrs);
        LayoutInflater inflater = LayoutInflater.from(context);
        RelativeLayout layout = (RelativeLayout) inflater.inflate(R.layout.level, null, false);
        LinearLayout linear = (LinearLayout)findViewById(R.id.levelMenu);
        linear.addView(layout);
    }


}
