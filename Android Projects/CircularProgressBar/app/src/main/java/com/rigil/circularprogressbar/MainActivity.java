package com.rigil.circularprogressbar;

import android.animation.ValueAnimator;
import android.os.Handler;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import com.budiyev.android.circularprogressbar.CircularProgressBar;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

    }


    @Override
    protected void onResume() {
        super.onResume();

        final CircularProgressBar foregroundProgressBar = findViewById(R.id.progress_bar_orange_foreground);

//        ValueAnimator progressAnimator = ValueAnimator.ofFloat(10, 100);
//        progressAnimator.addUpdateListener(new ValueAnimator.AnimatorUpdateListener() {
//            @Override
//            public void onAnimationUpdate(ValueAnimator animation) {
//                float animatedValue = (float) animation.getAnimatedValue();
//                System.out.println("Kien: " + animatedValue);
//                foregroundProgressBar.setProgress(animatedValue);
//                foregroundProgressBar.requestLayout();
//            }
//        });
//        progressAnimator.start();

        Button button = findViewById(R.id.button);
        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                foregroundProgressBar.setProgress(50f);
            }
        });


        final TextView progressText = findViewById(R.id.textView);

        final Handler handler = new Handler();
        // Define the code block to be executed
        Runnable runnableCode = new Runnable() {
            @Override
            public void run() {
                final float progress = 50f;
                foregroundProgressBar.setProgress(progress * 75 / 100);

                ValueAnimator progressAnimator = ValueAnimator.ofFloat(10, 100);
                progressAnimator.addUpdateListener(new ValueAnimator.AnimatorUpdateListener() {
                    @Override
                    public void onAnimationUpdate(ValueAnimator animation) {
                        float animatedValue = (float) animation.getAnimatedValue();
                        System.out.println("Kien: " + animatedValue);
                        progressText.setText(((int) progress) + "%");
                    }
                });

//                System.out.println("emiterX: " + emiterX + "\nemiterY: " + emiterY);
//                handler.postDelayed(this, 500);

                progressAnimator.start();
            }
        };
        // Run the above code block on the main thread after 2 seconds
        handler.postDelayed(runnableCode, 3000);







//        foregroundProgressBar.setProgress(50f);

    }

}
