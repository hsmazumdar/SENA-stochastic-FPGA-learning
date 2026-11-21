module class7_tree19(input wire [467:0] i, output wire [0:0] o);

reg [0:0] o_reg;
assign o = o_reg;

always @(*) begin
	if(((~i[197]) & i[355]) | (i[153] & (~i[355])) | ((~i[153]) & (~i[355])) | (i[121] & (~i[423]) & i[197] & i[355]) | (i[181] & i[231] & i[423] & i[197] & i[355]) | (i[184] & (~i[231]) & i[423] & i[197] & i[355]) | ((~i[184]) & (~i[231]) & i[423] & i[197] & i[355]) | (i[96] & (~i[121]) & (~i[423]) & i[197] & i[355]))
		o_reg = 0;
	else
		o_reg = 1;
end

endmodule
