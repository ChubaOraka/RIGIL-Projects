package com.rigil.particleeffect;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.app.Activity;
import android.graphics.Canvas;
import android.os.Bundle;
import android.os.Handler;
import android.view.View;
import android.view.View.OnClickListener;
import android.widget.FrameLayout;
import android.widget.ImageView;

public class MainActivity extends Activity implements OnClickListener {

    private final static int NUM_PARTICLES = 25;
    private final static int FRAME_RATE = 30;
    private final static int LIFETIME = 300;
    private Handler mHandler;
    private View mFX;
    private Explosion mExplosion;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        FrameLayout layout = (FrameLayout) findViewById(R.id.frame);
        final ImageView iv = (ImageView) findViewById(R.id.bomb_img);
        mFX = new View(this) {
            @Override
            protected void onDraw(Canvas c) {
                if (mExplosion!=null && !mExplosion.isDead())  {
                    mExplosion.update(c);
                    mHandler.removeCallbacks(mRunner);
                    mHandler.postDelayed(mRunner, FRAME_RATE);
                } else if (mExplosion!=null && mExplosion.isDead()) {
                    iv.setAlpha(1f);
                }
                super.onDraw(c);
            }
        };
        mFX.setLayoutParams(new FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT));
        layout.addView(mFX);
        mHandler = new Handler();
        iv.setOnClickListener(this);
    }

    @Override
    public void onClick(View v) {
        if (mExplosion==null || mExplosion.isDead()) {
            int[] loc = new int[2];
            v.getLocationOnScreen(loc);
            int offsetX = (int) (loc[0] + (v.getWidth()*.25));
            int offsetY = (int) (loc[1] - (v.getHeight()*.5));
            mExplosion = new Explosion(NUM_PARTICLES, offsetX, offsetY, this);
            mHandler.removeCallbacks(mRunner);
            mHandler.post(mRunner);
            v.animate().alpha(0).setDuration(LIFETIME).start();
        }
    }

    private Runnable mRunner = new Runnable() {
        @Override
        public void run() {
            mHandler.removeCallbacks(mRunner);
            mFX.invalidate();
        }
    };

}
