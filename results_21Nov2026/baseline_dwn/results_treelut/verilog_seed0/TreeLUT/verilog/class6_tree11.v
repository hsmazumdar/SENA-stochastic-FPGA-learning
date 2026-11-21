module class6_tree11(input wire [467:0] i, output wire [0:0] o);

reg [0:0] o_reg;
assign o = o_reg;

always @(*) begin
	if((i[423] & (~i[437])) | ((~i[423]) & (~i[437])) | ((~i[433]) & i[331] & i[437]) | (i[223] & (~i[331]) & i[437]) | (i[245] & (~i[387]) & i[433] & i[331] & i[437]))
		o_reg = 0;
	else
		o_reg = 1;
end

endmodule
